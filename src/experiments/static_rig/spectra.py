"""Full-record spectra of one stationary recording -> rotor speeds -> line powers.

The recording is a drone fixed in place with its rotors at constant speed, so
every rotor is a set of fixed lines at integer multiples of its shaft rate. The
whole motor-on span is read as one long-segment Welch spectrum per channel
(resolution ``1/seg_s`` Hz) and the channels are combined as the mean of their
line-over-floor prominence in dB.

Speeds are read in the SPEED domain: every spectral peak ``p`` near a harmonic
of the rig's comb rate ``f̄`` is a candidate ``p/j`` for each order ``j`` that
puts it within ``±spread_frac·f̄``, with an uncertainty ``~df/j``. One rotor
puts consistent candidates at every order, so its speed is a sharp mode of the
candidates' kernel density; high orders sharpen it (their ``df/j`` is small)
and rotors closer than the low orders can resolve separate there. A mode is
kept when at least ``min_support`` distinct orders vote for it.

Line powers are then summed in ``±line_bins`` bins around ``j·s`` per channel,
floor removed; a line within ``2·line_bins`` bins of another rotor's line at the
same order is flagged as merged rather than split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np


@dataclass(frozen=True)
class Params:
    #: Welch segment length (s): the frequency resolution is ``1/seg_s``.
    seg_s: float = 16.0
    f_max_hz: float = 8000.0
    #: Width of the running-median floor (Hz).
    floor_bw_hz: float = 20.0
    #: Motor-on detection: 0.5 s blocks within ``on_drop_db`` of the loud level.
    env_block_s: float = 0.5
    on_drop_db: float = 6.0
    trim_s: float = 1.0
    min_active_s: float = 8.0
    #: Comb (shaft-rate) search.
    lo_rev_s: float = 15.0
    hi_rev_s: float = 260.0
    comb_orders: int = 12
    comb_tol_frac: float = 0.03
    comb_step: float = 0.01
    #: ``s/m`` replaces ``s`` when the orders it adds carry lines this much
    #: (dB) over control positions half an order away.
    octave_db: float = 1.0
    #: Speed spectrum: grid step (rev/s), peak threshold over its median (dB),
    #: shoulder rule (a peak within ``merge_hwhm`` half-widths of a taller one
    #: is not a separate rotor), spectral peaks and orders for the refinement.
    spread_frac: float = 0.08
    kde_step: float = 0.0005
    h_min_db: float = 1.0
    #: ... and over this fraction of the tallest peak's height.
    h_rel: float = 0.25
    #: Weaker peaks (down to this, dB over the median) are listed as
    #: ``weak_candidates`` for inspection, never counted as rotors.
    h_weak_db: float = 0.3
    merge_hwhm: float = 2.0
    peak_db: float = 6.0
    min_support: int = 5
    #: Line-power window (bins either side of ``j·s``).
    line_bins: int = 4


# ─── segment + spectrum ──────────────────────────────────────────────────────


def motor_on_span(x_ct: np.ndarray, fs: int, p: Params) -> tuple[int, int]:
    """``(start, stop)`` samples of the longest loud span (rotors on), trimmed."""
    from scipy.signal import butter, sosfilt

    sos = butter(4, 40.0, "highpass", fs=fs, output="sos")
    y = np.asarray(sosfilt(sos, x_ct, axis=-1))
    blk = max(1, int(p.env_block_s * fs))
    n = y.shape[-1] // blk
    if n < 4:
        return 0, x_ct.shape[-1]
    e = (y[:, : n * blk].reshape(y.shape[0], n, blk) ** 2).mean(axis=(0, 2))
    lev = 10.0 * np.log10(e + 1e-20)
    on = lev >= np.percentile(lev, 90) - p.on_drop_db
    best, run, start, best_start = 0, 0, 0, 0
    for i, v in enumerate(on):
        if v:
            if run == 0:
                start = i
            run += 1
            if run > best:
                best, best_start = run, start
        else:
            run = 0
    trim = int(p.trim_s / p.env_block_s)
    a, b = best_start + trim, best_start + best - trim
    if (b - a) * p.env_block_s < p.min_active_s:
        return 0, x_ct.shape[-1]
    return a * blk, b * blk


@dataclass
class Spectrum:
    f: np.ndarray  # (F,)
    psd: np.ndarray  # (C, F) linear power density
    floor: np.ndarray  # (C, F) linear floor density
    df: float
    fs: int

    @property
    def prom(self) -> np.ndarray:
        """(C, F) line-over-floor prominence in dB."""
        return 10.0 * np.log10(self.psd / self.floor)

    @property
    def prom_mean(self) -> np.ndarray:
        """(F,) channel-mean prominence in dB."""
        return self.prom.mean(axis=0)


def _floor(p_db: np.ndarray, half: int) -> np.ndarray:
    """Running median of each row over ``2·half+1`` bins (strided, interpolated)."""
    from scipy.ndimage import median_filter

    stride = max(1, half // 16)
    sub = p_db[:, ::stride]
    width = max(3, 2 * (half // stride) + 1)
    med = median_filter(sub, size=(1, width), mode="nearest")
    x_full = np.arange(p_db.shape[1])
    x_sub = x_full[::stride]
    return np.stack([np.interp(x_full, x_sub, row) for row in med])


def spectrum(x_ct: np.ndarray, fs: int, p: Params) -> Spectrum:
    from scipy.signal import welch

    n = x_ct.shape[-1]
    nper = int(min(p.seg_s * fs, n // 2))
    f, pxx = welch(
        x_ct,
        fs=fs,
        window="hann",
        nperseg=nper,
        noverlap=nper // 2,
        detrend=cast(Any, False),
        axis=-1,
    )
    keep = (f > 0) & (f <= min(p.f_max_hz, 0.45 * fs))
    f, pxx = f[keep], np.maximum(pxx[:, keep], 1e-30)
    df = float(f[1] - f[0])
    p_db = 10.0 * np.log10(pxx)
    floor = 10.0 ** (_floor(p_db, max(2, round(0.5 * p.floor_bw_hz / df))) / 10.0)
    return Spectrum(f=f, psd=pxx, floor=floor, df=df, fs=int(fs))


# ─── comb rate ────────────────────────────────────────────────────────────────


class _RangeMax:
    """O(1) range-maximum queries over a fixed array (sparse table)."""

    def __init__(self, a: np.ndarray) -> None:
        self.levels = [np.asarray(a, dtype=np.float64)]
        k = 1
        while 2 * k <= len(a):
            prev = self.levels[-1]
            self.levels.append(np.maximum(prev[:-k], prev[k:]))
            k *= 2

    def query(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """max over ``[lo, hi)``; empty ranges give ``-inf``."""
        n = len(self.levels[0])
        lo = np.clip(lo, 0, n)
        hi = np.clip(hi, 0, n)
        out = np.full(lo.shape, -np.inf)
        ok = hi > lo
        length = np.where(ok, hi - lo, 1)
        lev = np.floor(np.log2(length)).astype(int)
        for L in np.unique(lev[ok]):
            sel = ok & (lev == L)
            tab = self.levels[L]
            out[sel] = np.maximum(tab[lo[sel]], tab[hi[sel] - (1 << L)])
        return out


def _band_max(sp: Spectrum, rmq: _RangeMax, centre: np.ndarray, half_hz: np.ndarray) -> np.ndarray:
    half = np.maximum(half_hz, 2.0 * sp.df)
    lo = np.searchsorted(sp.f, centre - half)
    hi = np.searchsorted(sp.f, centre + half, side="right")
    return rmq.query(lo, hi)


def comb_rate(sp: Spectrum, p: Params) -> dict[str, Any]:
    """The rig's dominant comb rate: best mean line prominence over orders
    ``1..comb_orders``. For a two- or three-bladed rotor this can be the
    blade-pass rate (2× / 3× the shaft); :func:`rotor_speeds` decides."""
    pm = sp.prom_mean
    rmq = _RangeMax(pm)
    grid = np.arange(p.lo_rev_s, p.hi_rev_s, p.comb_step)
    score = np.zeros_like(grid)
    count = np.zeros_like(grid)
    for k in range(1, p.comb_orders + 1):
        c = k * grid
        ok = c < sp.f[-1]
        m = _band_max(sp, rmq, c[ok], p.comb_tol_frac * c[ok])
        score[ok] += np.clip(m, 0.0, 30.0)
        count[ok] += 1
    score = np.where(count > 0, score / np.maximum(count, 1), 0.0)
    return {"f0_comb": float(grid[int(np.argmax(score))]), "score_db": float(score.max())}


def octave_evidence(sp: Spectrum, speeds: list[float], width: float, m: int) -> float:
    """Mean prominence (dB, clipped 0..30) at the orders ``j·s/m`` that a
    shaft rate ``s/m`` ADDS (``j`` not a multiple of ``m``), minus the same at
    control positions half an order away, over every rotor speed ``s``. The
    positions are exact (refined speeds), so a multi-rotor rig's other lines
    do not leak in."""
    pm = np.clip(sp.prom_mean, 0.0, 30.0)
    rmq = _RangeMax(pm)
    added, control = [], []
    for s in speeds:
        base = s / m
        js = np.array([j for j in range(1, int(sp.f[-1] // base)) if j % m], dtype=float)
        if js.size == 0:
            continue
        tol = np.maximum(2.0 * sp.df, 2.0 * js * width / m)
        added.append(_band_max(sp, rmq, js * base, tol))
        control.append(_band_max(sp, rmq, (js + 0.5) * base, tol))
    if not added:
        return 0.0
    return float(np.mean(np.concatenate(added)) - np.mean(np.concatenate(control)))


def rotor_speeds(sp: Spectrum, n_rotors: int, p: Params) -> dict[str, Any]:
    """Comb rate → rotor speeds around it → octave decision on those exact
    speeds (``s/m`` for m = 2, 3 wins when the orders it adds carry lines
    ``octave_db`` over the controls, median over the rotors) → speeds re-read
    around the shaft rate."""
    comb = comb_rate(sp, p)
    modes = speed_modes(sp, comb["f0_comb"], n_rotors, p)
    speeds = [r["speed"] for r in modes["rotors"]]
    width = max([r["hwhm"] for r in modes["rotors"]] + [p.kde_step])
    comb["octave"] = 1
    for m in (2, 3):
        if not speeds or comb["f0_comb"] / m < p.lo_rev_s:
            continue
        ev = float(np.median([octave_evidence(sp, [s], width, m) for s in speeds]))
        comb[f"octave_evidence_db_{m}"] = ev
        if ev >= p.octave_db:
            comb["octave"] = m
            break
    comb["f_bar"] = comb["f0_comb"] / comb["octave"]
    if comb["octave"] != 1:
        modes = speed_modes(sp, comb["f_bar"], n_rotors, p)
    return {"comb": comb, **modes}


# ─── speed-domain modes ──────────────────────────────────────────────────────


def _peaks(
    sp: Spectrum, lo_hz: float, hi_hz: float, min_db: float
) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of the channel-mean prominence in ``[lo, hi]`` above
    ``min_db``: parabolic-interpolated frequency and height (dB)."""
    pm = sp.prom_mean
    i0 = max(1, int(np.searchsorted(sp.f, lo_hz)))
    i1 = min(len(pm) - 1, int(np.searchsorted(sp.f, hi_hz)))
    seg = pm[i0 - 1 : i1 + 1]
    mid = seg[1:-1]
    is_pk = (mid > seg[:-2]) & (mid >= seg[2:]) & (mid >= min_db)
    idx = np.nonzero(is_pk)[0] + i0
    a, b, c = pm[idx - 1], pm[idx], pm[idx + 1]
    den = a - 2 * b + c
    off = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / den, 0.0)
    off = np.clip(off, -0.5, 0.5)
    return sp.f[idx] + off * sp.df, b - 0.25 * (a - c) * off


def _assign(
    pk_f: np.ndarray, pk_h: np.ndarray, speeds: np.ndarray, df: float, width: float
) -> list[dict[int, tuple[float, float]]]:
    """One-to-one peak assignment: every peak goes to the rotor whose nearest
    harmonic ``j·s`` it matches best (within ``max(1.5·df, 2·j·width)``), and
    every (rotor, order) keeps only its highest peak. Returns, per rotor,
    ``{order: (peak Hz, height dB)}``."""
    out: list[dict[int, tuple[float, float]]] = [{} for _ in speeds]
    for pf, ph in zip(pk_f, pk_h, strict=True):
        best = None
        for r, s in enumerate(speeds):
            j = int(round(pf / s))
            if j < 1:
                continue
            tau = max(1.5 * df, 2.0 * j * width)
            z = abs(pf - j * s) / tau
            if z < 1.0 and (best is None or z < best[0]):
                best = (z, r, j)
        if best is None:
            continue
        _, r, j = best
        if j not in out[r] or out[r][j][1] < ph:
            out[r][j] = (float(pf), float(ph))
    return out


def _fit(assigned: dict[int, tuple[float, float]]) -> tuple[float, float, float]:
    """``(speed, se, scatter)`` from ``p ≈ j·s`` weighted by line height."""
    j = np.array(list(assigned), dtype=float)
    pf = np.array([v[0] for v in assigned.values()])
    w = np.clip(np.array([v[1] for v in assigned.values()]), 0.1, None)
    s = float(np.sum(w * j * pf) / np.sum(w * j * j))
    r = pf / j - s
    scatter = float(np.sqrt(np.sum(w * r * r) / np.sum(w)))
    return s, scatter / math.sqrt(max(1, j.size - 1)), scatter


def speed_spectrum(
    sp: Spectrum, grid: np.ndarray, skip_multiple: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``H(s)``: the channel-mean prominence (dB, clipped to 0..30) at ``j·s``
    averaged over every order ``j`` the band holds, and the order count.

    One rotor at ``s`` is a peak of ``H`` whose width is the larger of its
    shaft wander and ``df/j``: wander widens order ``j`` by ``j`` in Hz, which
    maps back to the same width in rev/s, while the resolution in rev/s
    improves as ``df/j`` — so many orders sharpen and stabilise the peak."""
    pm = np.clip(sp.prom_mean, 0.0, 30.0)
    acc = np.zeros_like(grid)
    n = np.zeros_like(grid)
    k_max = int(sp.f[-1] // grid[-1])
    for j in range(1, k_max + 1):
        if skip_multiple is not None and j % skip_multiple == 0:
            continue
        acc += np.interp(j * grid, sp.f, pm)
        n += 1
    return acc / np.maximum(n, 1), n


def _peak_hwhm(h: np.ndarray, i: int, step: float) -> float:
    half = 0.5 * (h[i] + np.median(h))
    lo = i
    while lo > 0 and h[lo] > half:
        lo -= 1
    hi = i
    while hi < h.size - 1 and h[hi] > half:
        hi += 1
    return 0.5 * (hi - lo) * step


def speed_modes(sp: Spectrum, f_bar: float, n_rotors: int, p: Params) -> dict[str, Any]:
    """Rotor speeds near ``f_bar`` from the speed spectrum ``H(s)``.

    Peaks of ``H`` above ``median + h_min_db`` are taken in height order; a
    peak within ``merge_hwhm`` half-widths of a taller one is its shoulder,
    not a rotor. At most ``n_rotors`` survive. Each is then refined by a
    one-to-one assignment of spectral peaks to (rotor, order) and a weighted
    ``p ≈ j·s`` fit: ``se`` is its formal error, ``scatter`` the per-order
    spread of ``p/j`` (rev/s), ``support`` the number of orders used.
    ``n_distinct < n_rotors`` means some rotors run at speeds the record
    cannot separate."""
    W = p.spread_frac * f_bar
    grid = np.arange(f_bar - W, f_bar + W, p.kde_step)
    h, _ = speed_spectrum(sp, grid)
    med = float(np.median(h))
    is_pk = np.r_[False, (h[1:-1] > h[:-2]) & (h[1:-1] >= h[2:]), False]
    strong = med + max(p.h_min_db, p.h_rel * (float(h.max()) - med))
    cand = [int(i) for i in np.nonzero(is_pk & (h >= strong))[0]]
    weak = [
        {"speed_h": float(grid[i]), "h_db": float(h[i] - med)}
        for i in np.nonzero(is_pk & (h >= med + p.h_weak_db) & (h < strong))[0]
    ]
    cand.sort(key=lambda i: -h[i])
    chosen: list[tuple[int, float]] = []
    for i in cand:
        hw = _peak_hwhm(h, i, p.kde_step)
        if all(abs(grid[i] - grid[c]) > p.merge_hwhm * max(hw, chw) for c, chw in chosen):
            chosen.append((i, hw))
        if len(chosen) == n_rotors:
            break
    k_max = int(sp.f[-1] // (f_bar + W))
    pk_f, pk_h = _peaks(sp, (f_bar - W) * 0.999, k_max * (f_bar + W), p.peak_db)
    speeds = np.array([grid[i] for i, _ in chosen])
    width = max([hw for _, hw in chosen] + [p.kde_step])
    rotors = []
    if speeds.size:
        for (i, hw), a in zip(chosen, _assign(pk_f, pk_h, speeds, sp.df, width), strict=True):
            rec: dict[str, Any] = {
                "speed_h": float(grid[i]),
                "h_db": float(h[i] - med),
                "hwhm": hw,
                "support": len(a),
                "orders": sorted(a),
            }
            if len(a) >= p.min_support:
                speed, se, scatter = _fit(a)
                rec.update(speed=speed, se=se, scatter=scatter)
            else:
                rec.update(speed=float(grid[i]), se=None, scatter=None)
            rotors.append(rec)
    rotors.sort(key=lambda d: d["speed"])
    return {
        "rotors": rotors,
        "n_distinct": len(rotors),
        "weak_candidates": weak,
        "n_rotors": n_rotors,
        "k_max": k_max,
        "h_median_db": med,
        "speed_spectrum": {"s0": float(grid[0]), "step": p.kde_step, "h_db": h.tolist()},
    }


# ─── line powers ─────────────────────────────────────────────────────────────


def line_powers(sp: Spectrum, speeds: list[float], p: Params) -> list[dict[str, Any]]:
    """Per rotor: per-order, per-channel line power (floor removed) and SNR.

    ``power_db[j-1][c]`` is ``10·log10(Σ psd·df − Σ floor·df)`` over the
    ``±line_bins`` window around ``j·s`` (NaN when the line is below its floor),
    ``snr_db`` the window's total over its floor, ``merged[j-1]`` true when
    another rotor's line at the same order falls within ``2·line_bins`` bins."""
    w_hz = p.line_bins * sp.df
    out = []
    for i, s in enumerate(speeds):
        k_max = int(sp.f[-1] // s)
        power = np.full((k_max, sp.psd.shape[0]), np.nan)
        snr = np.full((k_max, sp.psd.shape[0]), np.nan)
        merged = []
        for k in range(1, k_max + 1):
            c = k * s
            lo = int(np.searchsorted(sp.f, c - w_hz))
            hi = int(np.searchsorted(sp.f, c + w_hz, side="right"))
            tot = sp.psd[:, lo:hi].sum(axis=1) * sp.df
            flo = sp.floor[:, lo:hi].sum(axis=1) * sp.df
            line = tot - flo
            power[k - 1] = np.where(line > 0, 10.0 * np.log10(np.maximum(line, 1e-30)), np.nan)
            snr[k - 1] = 10.0 * np.log10(tot / flo)
            merged.append(any(abs(k * o - c) < 2 * w_hz for jj, o in enumerate(speeds) if jj != i))
        out.append({"speed": float(s), "power_db": power, "snr_db": snr, "merged": merged})
    return out


def comb_band_powers(
    sp: Spectrum, s_lo: float, s_hi: float, p: Params, flank_hz: float = 10.0
) -> dict[str, Any]:
    """Total (all-rotor) comb power per order and channel, for rotors that
    cannot be told apart: every rotor runs somewhere in ``[s_lo, s_hi]``.

    Order ``k`` integrates ``[k·s_lo − w, k·s_hi + w]`` (``w = line_bins·df``)
    minus its floor, the per-channel median density of the two ``flank_hz``
    flanks just outside the band (the running-median floor is biased once the
    band fills its window). Orders stop where the band plus flanks would reach
    the next order's band, so no line is counted twice or used as floor."""
    w = p.line_bins * sp.df
    orders, power, snr = [], [], []
    k = 1
    while True:
        lo_hz, hi_hz = k * s_lo - w, k * s_hi + w
        if hi_hz + flank_hz > sp.f[-1] or s_lo - k * (s_hi - s_lo) < 2 * (w + flank_hz):
            break
        band = (sp.f >= lo_hz) & (sp.f <= hi_hz)
        flanks = ((sp.f >= lo_hz - flank_hz) & (sp.f < lo_hz)) | (
            (sp.f > hi_hz) & (sp.f <= hi_hz + flank_hz)
        )
        tot = sp.psd[:, band].sum(axis=1) * sp.df
        flo = np.median(sp.psd[:, flanks], axis=1) * int(band.sum()) * sp.df
        line = tot - flo
        orders.append(k)
        power.append(np.where(line > 0, 10.0 * np.log10(np.maximum(line, 1e-30)), np.nan))
        snr.append(10.0 * np.log10(tot / flo))
        k += 1
    return {
        "orders": orders,
        "power_db": np.asarray(power).reshape(len(orders), sp.psd.shape[0]),
        "snr_db": np.asarray(snr).reshape(len(orders), sp.psd.shape[0]),
    }
