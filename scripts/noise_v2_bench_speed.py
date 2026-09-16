#!/usr/bin/env python3
"""One rotor speed per rotor for every static/bench drone-noise recording.

Survey instrument of the *noise model v2* corpus pass: it turns a stationary
recording into a **fit point** — one shaft speed per rotor, a tolerance, and a
usable/rejected verdict — with a simple, auditable, label-free estimator.

The estimator (per recording)

1. **Stationary window.** 1 s RMS blocks; the longest run of blocks inside
   ``--level-tol-db`` of the run median is the stationary part; the window is
   its centre, ``--window-s`` long (at least ``--min-window-s``).
2. **Welch spectrum.** Per channel, median-averaged, ``nperseg`` chosen for a
   target resolution of ``--target-df-hz``; the per-channel power spectra are
   averaged (incoherent — no beamforming), then converted to dB. The broadband
   floor is a wide running median of the dB spectrum.
3. **Harmonic-sum score.** For a shaft-rate candidate ``f`` (rev/s == Hz of
   shaft rotation) the score is the mean over harmonics ``k = 1..K`` of the
   line prominence over the floor inside a narrow band at ``k·f``, clipped to
   ``--prom-clip-db``. Candidates run over ``--lo``..``--hi`` rev/s on a coarse
   grid, then a local parabolic refinement.
4. **Odd-harmonic octave check.** The DREGON-style failure is a *doubling*: a
   two-bladed rotor puts most energy on the blade-pass line at ``2·f``, so the
   argmax lands there. The half ``f/2`` is accepted only when its ODD
   harmonics (``k = 1, 3, 5, …`` — the lines that do not exist if the argmax is
   already the shaft rate) clear the floor by ``--odd-margin-db`` on at least
   ``--odd-min-frac`` of the tested orders. Verdict is reported per recording
   (``as_found`` / ``halved`` / ``halved_twice``).
5. **Per-rotor split.** At high harmonic order the four rotor combs separate.
   Peaks within ±``--split-frac`` of ``k·f`` are collected at several orders;
   the order with the most resolved peaks gives the per-rotor fundamentals and
   the spread. One peak → one speed (a single-rotor bench, or four rotors the
   spectrum cannot separate).

**Tolerance rule (applied, not only stated).** A recording is ``usable`` when

* the harmonic-sum peak clears the best candidate OUTSIDE its own octave family
  (``f/2``, ``f``, ``2·f``) by ``--min-margin-db`` (default 3 dB), and
* the reading is stable to ``--max-half-delta`` (default 1 rev/s) between two
  disjoint halves of the window, and
* the window is at least ``--min-window-s`` long (default 8 s).

The reported tolerance is ``max(half_delta, 0.5·df_rev_s, 0.25)`` rev/s.

Cross-checks written into the output: the DREGON throttle law
(``rate = 0.975·throttle + 0.37`` rev/s), the SPCUP19 AGH single-rotor blind
readings, the KAIST stated 3010 RPM, and the DroneAudioSet paper's stated
spectral lines (168/235 Hz for D_large, 156/259 Hz for D_small).

Run (laptop smoke, then the corpus-wide pass on uni-cpu):

  PYTHONPATH=src python scripts/noise_v2_bench_speed.py --corpora dregon_bench --limit 2
  PYTHONPATH=src python scripts/noise_v2_bench_speed.py --corpora all
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))

# ── Frozen external references (not recomputed here) ─────────────────────────

#: Validated DREGON single-motor bench throttle law (blind-corpus-annotation.md).
DREGON_LAW = (0.975, 0.37)

#: Accepted SPCUP19 AGH single-rotor blind readings, rev/s, by take index
#: (results/blind_corpus/spcup_single/report/by_recording.csv).
SPCUP_AGH_BLIND = {
    "0": 79.71,
    "1": 66.33,
    "2": 113.78,
    "3": 97.43,
    "4": 77.74,
    "5": 97.81,
    "6": 96.59,
    "7": 96.81,
}

#: KAIST rotating-machine dataset-stated shaft rate (3010 RPM).
KAIST_NOMINAL_REV_S = 3010.0 / 60.0

#: DroneAudioSet paper (arXiv:2510.15383, Fig. 7) stated spectral lines, Hz.
DASET_PAPER_LINE_HZ = {
    ("D_large", "low"): 168.0,
    ("D_large", "high"): 235.0,
    ("D_small", "low"): 156.0,
    ("D_small", "high"): 259.0,
}

#: Rotor counts per corpus/condition family.
N_ROTORS = {
    "dregon_bench_single": 1,
    "dregon_bench_all": 4,
    "spcup_single_rotor": 1,
    "spcup_quad": 4,
    "daset": 4,
    "kaist": 1,
    "drone_audio": 4,
    "zenodo": 4,
    "chums_bench": 1,
}


# ── Estimator configuration ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Config:
    lo: float = 15.0
    #: detection scans lo..hi*this, so the blade-pass comb of a `hi` rotor is in range
    hi_detect_mult: float = 2.0
    hi: float = 150.0
    window_s: float = 16.0
    min_window_s: float = 8.0
    level_tol_db: float = 4.0
    active_range_db: float = 12.0
    min_welch_segments: int = 4
    target_df_hz: float = 0.5
    f_max_hz: float = 3000.0
    k_max: int = 40
    prom_clip_db: float = 30.0
    floor_bw_hz: float = 60.0
    band_frac: float = 0.003
    coarse_step: float = 0.02
    odd_margin_db: float = 6.0
    odd_min_frac: float = 0.5
    odd_orders: tuple[int, ...] = (1, 3, 5, 7, 9, 11)
    min_margin_db: float = 3.0
    max_half_delta: float = 1.0
    split_frac: float = 0.05
    split_orders: tuple[int, ...] = (8, 12, 16, 20, 24, 28, 32)
    split_prom_db: float = 6.0
    max_channels: int = 8


# ── Spectrum primitives ──────────────────────────────────────────────────────


def _floor_median(y: np.ndarray, half: int, stride: int = 8) -> np.ndarray:
    """Broadband floor: median filter on a strided subsample, interpolated back.

    The floor is smooth by construction (``--floor-bw-hz`` wide), so filtering
    every 8th bin and interpolating is indistinguishable from the full running
    median and ~8x cheaper on a 65 k-bin spectrum.
    """
    from scipy.ndimage import median_filter

    if half < 1:
        return y.copy()
    idx = np.arange(0, y.size, stride)
    sub = median_filter(y[idx], size=max(3, 2 * (half // stride) + 1), mode="nearest")
    return np.interp(np.arange(y.size), idx, sub)


def welch_db(x_ct: np.ndarray, fs: int, cfg: Config) -> tuple[np.ndarray, np.ndarray, float]:
    """``(freqs, power_db, df)`` — channel-averaged median Welch spectrum in dB."""
    from scipy.signal import welch

    x = np.asarray(x_ct, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    x = x[: cfg.max_channels]
    n = x.shape[-1]
    # Resolution target, but never fewer than `min_welch_segments` half-overlapped
    # segments: a 6 s half-window must still average, or a single noisy bin wins.
    nper = int(2 ** math.ceil(math.log2(max(64.0, fs / cfg.target_df_hz))))
    nper = min(nper, int(2 ** math.floor(math.log2(max(256, 2 * n // cfg.min_welch_segments)))))
    nper = min(nper, n)
    f, p = welch(
        x,
        fs=fs,
        nperseg=nper,
        noverlap=nper // 2,
        window="hann",
        average="median" if n >= 3 * nper else "mean",
        detrend="constant",
        axis=-1,
    )
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 2:
        p = p.mean(axis=0)
    p = np.maximum(p, 1e-30)
    return f, 10.0 * np.log10(p), float(f[1] - f[0])


def _floor_db(p_db: np.ndarray, df: float, cfg: Config) -> np.ndarray:
    return _floor_median(p_db, max(1, int(round(0.5 * cfg.floor_bw_hz / max(df, 1e-9)))))


@dataclass
class Spectrum:
    f: np.ndarray
    p_db: np.ndarray
    floor: np.ndarray
    df: float
    fs: int
    prom: np.ndarray = field(init=False)
    _maxf: dict[int, np.ndarray] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.prom = self.p_db - self.floor

    def max_filtered(self, width: int) -> np.ndarray:
        """``prom`` maximum-filtered over ``width`` bins (cached per width)."""
        from scipy.ndimage import maximum_filter1d

        width = int(max(1, width | 1))
        if width not in self._maxf:
            self._maxf[width] = (
                self.prom if width == 1 else maximum_filter1d(self.prom, size=width, mode="nearest")
            )
        return self._maxf[width]

    def band_prom_at(self, centres: np.ndarray, half_bins: np.ndarray) -> np.ndarray:
        """Vectorised band maxima of ``prom`` at ``centres`` Hz, ± ``half_bins``."""
        idx = np.clip(np.rint(centres / self.df).astype(np.int64), 0, self.prom.size - 1)
        widths = 2 * np.maximum(1, np.ceil(half_bins).astype(np.int64)) + 1
        out = np.empty(centres.shape, dtype=np.float64)
        for w in np.unique(widths):
            sel = widths == w
            out[sel] = self.max_filtered(int(w))[idx[sel]]
        return out

    def band_prom(self, centre: float, half_hz: float) -> tuple[float, float]:
        """``(peak prominence dB, peak frequency)`` in ``centre ± half_hz``."""
        lo = int(max(0, math.floor((centre - half_hz) / self.df)))
        hi = int(min(self.prom.size - 1, math.ceil((centre + half_hz) / self.df)))
        if hi <= lo:
            return (-99.0, centre)
        seg = self.prom[lo : hi + 1]
        j = int(np.argmax(seg))
        return (float(seg[j]), float(self.f[lo + j]))


def make_spectrum(x_ct: np.ndarray, fs: int, cfg: Config) -> Spectrum:
    f, p_db, df = welch_db(x_ct, fs, cfg)
    return Spectrum(f=f, p_db=p_db, floor=_floor_db(p_db, df, cfg), df=df, fs=fs)


# ── Harmonic-sum score ───────────────────────────────────────────────────────


def n_harmonics(f0: float, sp: Spectrum, cfg: Config) -> int:
    f_top = min(cfg.f_max_hz, 0.45 * sp.fs)
    return int(max(2, min(cfg.k_max, math.floor(f_top / max(f0, 1e-6)))))


def score_many(f0s: np.ndarray, sp: Spectrum, cfg: Config) -> np.ndarray:
    """Mean clipped line prominence over the harmonics of every ``f0`` (dB).

    One vector pass per harmonic order: the band maximum at ``k·f0`` comes from
    a cached maximum-filtered copy of the prominence spectrum, so the whole
    15–150 rev/s grid costs ``k_max`` numpy gathers, not ``N·k_max`` slices.
    """
    f = np.atleast_1d(np.asarray(f0s, dtype=np.float64))
    f_top = min(cfg.f_max_hz, 0.45 * sp.fs)
    k_hi = np.clip(np.floor(f_top / np.maximum(f, 1e-6)), 2, cfg.k_max)
    acc = np.zeros(f.shape)
    cnt = np.zeros(f.shape)
    for k in range(1, cfg.k_max + 1):
        sel = k_hi >= k
        if not np.any(sel):
            break
        centres = k * f[sel]
        half_bins = np.maximum(1.5, cfg.band_frac * centres / sp.df)
        vals = sp.band_prom_at(centres, half_bins)
        acc[sel] += np.clip(vals, 0.0, cfg.prom_clip_db)
        cnt[sel] += 1.0
    return acc / np.maximum(cnt, 1.0)


def score(f0: float, sp: Spectrum, cfg: Config) -> float:
    return float(score_many(np.array([f0]), sp, cfg)[0])


def score_grid(sp: Spectrum, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(cfg.lo, cfg.hi + 1e-9, cfg.coarse_step)
    return grid, score_many(grid, sp, cfg)


def _refine(f0: float, sp: Spectrum, cfg: Config) -> tuple[float, float]:
    """Local refinement of the score peak around ``f0`` (halving step search)."""
    step = cfg.coarse_step
    best_f, best_v = f0, score(f0, sp, cfg)
    for _ in range(6):
        step *= 0.5
        cands = np.array([best_f - step, best_f + step])
        vals = score_many(cands, sp, cfg)
        j = int(np.argmax(vals))
        if vals[j] > best_v:
            best_f, best_v = float(cands[j]), float(vals[j])
    return best_f, best_v


def _local_maxima(vals: np.ndarray) -> np.ndarray:
    up = np.r_[True, vals[1:] > vals[:-1]]
    dn = np.r_[vals[:-1] >= vals[1:], True]
    return np.where(up & dn)[0]


#: Small-rational relatives of a comb: a candidate at ``m/n · f`` scores on a
#: sub- or superset of ``f``'s own teeth, so it is the SAME family, not a rival.
#: ``2·f`` (the blade-pass doubling) and ``f/2`` are the octave members.
_FAMILY_RATIOS = (0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75, 1.0, 4.0 / 3.0, 1.5, 2.0, 3.0, 4.0)
_OCTAVE_RATIOS = (0.5, 1.0, 2.0)


def _in_family(f: float, ref: float, ratios: tuple[float, ...], tol: float = 0.03) -> bool:
    return any(abs(f / (r * ref) - 1.0) <= tol for r in ratios)


def odd_margin(half: float, sp: Spectrum, cfg: Config) -> tuple[float, float, list[float]]:
    """Evidence that ``half`` (not ``2·half``) is the shaft rate.

    Returns ``(median odd prominence dB, fraction of orders over the margin,
    per-order prominences)`` over the ODD harmonics of ``half``.
    """
    proms: list[float] = []
    for k in cfg.odd_orders:
        centre = k * half
        if centre > min(cfg.f_max_hz, 0.45 * sp.fs):
            break
        prom, _ = sp.band_prom(centre, max(1.5 * sp.df, cfg.band_frac * centre))
        proms.append(float(prom))
    if not proms:
        return (-99.0, 0.0, [])
    arr = np.array(proms)
    return (float(np.median(arr)), float(np.mean(arr >= cfg.odd_margin_db)), proms)


def shaft_rate(sp: Spectrum, cfg: Config) -> dict[str, Any]:
    """Shaft rate of a stationary spectrum, with the octave decision.

    Three quantities, because a two-bladed rotor puts most of its energy on the
    blade-pass line at twice the shaft rate:

    * **Detection.** The harmonic-sum argmax over ``lo``..``hi`` rev/s. The
      range is the physical rotor-speed prior; widening it would let a
      sparse superharmonic comb (every third tooth of the truth) win by
      averaging fewer, stronger teeth.
    * **Octave mapping.** Halve while the ODD harmonics of the half clear
      ``--odd-margin-db`` on ``--odd-min-frac`` of the tested orders: direct
      evidence that the half, not the detected rate, is the shaft.
    * **Margin.** Measured on the *comb family* score
      ``max(score(f), score(2·f))`` — the blade-pass reading of the same rotor
      — for the accepted rate and for every rival alike. Without this the
      margin of a blade-pass-dominated rig is diluted by its own missing odd
      teeth (DroneAudioSet: 7.4 dB at the shaft rate against 12.0 dB at the
      blade-pass line of the same rotor).
    """
    grid = np.arange(cfg.lo, cfg.hi + 1e-9, cfg.coarse_step)
    vals = score_many(grid, sp, cfg)
    fam = np.maximum(vals, score_many(2.0 * grid, sp, cfg))
    f_det, s_det = _refine(float(grid[int(np.argmax(vals))]), sp, cfg)
    s_fam = max(s_det, score(2.0 * f_det, sp, cfg))

    verdict, halvings = "as_found", 0
    f_final = f_det
    trace: list[dict[str, Any]] = []
    while halvings < 2:
        half = f_final / 2.0
        if half < cfg.lo:
            break
        med, frac, proms = odd_margin(half, sp, cfg)
        odd_ok = med >= cfg.odd_margin_db and frac >= cfg.odd_min_frac
        trace.append(
            {
                "half_rev_s": round(half, 4),
                "odd_median_db": round(med, 3),
                "odd_frac": round(frac, 3),
                "odd_prom_db": [round(p, 2) for p in proms],
                "odd_evidence": bool(odd_ok),
            }
        )
        if not odd_ok:
            break
        f_final, _ = _refine(half, sp, cfg)
        halvings += 1
        verdict = "halved" if halvings == 1 else "halved_twice"
    s_final = score(f_final, sp, cfg)

    # Runner-up: the strongest comb-family score that is NOT a small-rational
    # relative of the detected rate (nor of the mapped shaft rate). The
    # octave-only exclusion of the written rule is reported alongside — on a
    # clean two-bladed bench the literal version's runner-up is the third
    # sub/super-harmonic, the same comb read every third tooth, not a rival.
    maxima = _local_maxima(fam)
    runner_f, runner_v = float("nan"), -99.0
    oct_f, oct_v = float("nan"), -99.0
    refs = (f_det, f_final)
    for idx in maxima:
        f_c = float(grid[idx])
        v_c = float(fam[idx])
        if not any(_in_family(f_c, r, _OCTAVE_RATIOS) for r in refs) and v_c > oct_v:
            oct_f, oct_v = f_c, v_c
        if not any(_in_family(f_c, r, _FAMILY_RATIOS) for r in refs) and v_c > runner_v:
            runner_f, runner_v = f_c, v_c
    margin = s_fam - runner_v if runner_v > -99.0 else float("inf")
    margin_oct = s_fam - oct_v if oct_v > -99.0 else float("inf")

    return {
        "rate_rev_s": f_final,
        "score_db": s_final,
        "detected_rev_s": f_det,
        "detected_score_db": s_det,
        "family_score_db": s_fam,
        "halvings": halvings,
        "octave_verdict": verdict,
        "octave_trace": trace,
        "runner_up_rev_s": runner_f,
        "runner_up_score_db": runner_v,
        "margin_db": margin,
        "octave_only_runner_up_rev_s": oct_f,
        "margin_octave_only_db": margin_oct,
    }


def per_rotor(sp: Spectrum, f0: float, n_rotors: int, cfg: Config) -> dict[str, Any]:
    """Per-rotor fundamentals from line splitting at high harmonic order."""
    best: dict[str, Any] = {"n_distinct": 1, "order": None, "speeds": [f0], "spread_rev_s": 0.0}
    for k in cfg.split_orders:
        centre = k * f0
        if centre > min(cfg.f_max_hz, 0.45 * sp.fs):
            break
        half = cfg.split_frac * centre
        lo = int(max(0, math.floor((centre - half) / sp.df)))
        hi = int(min(sp.prom.size - 1, math.ceil((centre + half) / sp.df)))
        if hi - lo < 8:
            continue
        seg = sp.prom[lo : hi + 1]
        idx = _local_maxima(seg)
        idx = idx[seg[idx] >= cfg.split_prom_db]
        if idx.size == 0:
            continue
        # keep the n_rotors strongest, merge peaks closer than 0.15 rev/s at k
        cand = sorted((float(sp.f[lo + i]) / k, float(seg[i])) for i in idx)
        merged: list[tuple[float, float]] = []
        for f_c, v in cand:
            if merged and abs(f_c - merged[-1][0]) < 0.15:
                if v > merged[-1][1]:
                    merged[-1] = (f_c, v)
                continue
            merged.append((f_c, v))
        merged.sort(key=lambda t: -t[1])
        keep = sorted(f for f, _ in merged[:n_rotors])
        if len(keep) > best["n_distinct"]:
            best = {
                "n_distinct": len(keep),
                "order": k,
                "speeds": [round(f, 4) for f in keep],
                "spread_rev_s": round(max(keep) - min(keep), 4),
            }
    return best


def _stationary_window(x_ct: np.ndarray, fs: int, cfg: Config) -> tuple[int, int, float]:
    """``(start, stop, active_s)`` — the stationary, motor-ON part of a recording.

    Blocks more than ``--active-range-db`` below the loudest block are the
    motor-off head/tail (a DREGON bench run is ~12 s of motor inside a 25 s
    file, and the silent tail carries a fixed 88 rev/s room tone that the
    harmonic sum will happily lock onto). The window is the longest run of
    ACTIVE blocks within ``--level-tol-db`` of the active median, truncated to
    ``--window-s`` around its centre; a shorter run yields a shorter window,
    which the ``--min-window-s`` gate then judges.
    """
    n = x_ct.shape[-1]
    blk = fs
    nb = n // blk
    want = int(round(cfg.window_s * fs))
    if nb < 2:
        return 0, n, n / fs
    mono = x_ct[0] if x_ct.ndim == 2 else x_ct
    rms = np.sqrt(np.maximum((mono[: nb * blk].reshape(nb, blk) ** 2).mean(axis=1), 1e-20))
    lvl = 20.0 * np.log10(rms)
    # 3-block median smoothing: an unsteady bench motor (DREGON Motor2 swings
    # 5 dB block to block at a fixed throttle) must not fragment its own run,
    # while a ramp block stays 6+ dB off the active median and is dropped.
    if nb >= 3:
        from scipy.ndimage import median_filter

        lvl_s = median_filter(lvl, size=3, mode="nearest")
    else:
        lvl_s = lvl
    active = lvl_s >= float(lvl_s.max()) - cfg.active_range_db
    med = float(np.median(lvl_s[active]))
    ok = active & (np.abs(lvl_s - med) <= cfg.level_tol_db)
    best_i, best_len, i = 0, 0, 0
    while i < nb:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < nb and ok[j]:
            j += 1
        if j - i > best_len:
            best_i, best_len = i, j - i
        i = j
    if best_len == 0:  # no stationary run at all: centre of the file
        start = max(0, (n - want) // 2)
        return start, min(n, start + want), 0.0
    run_lo, run_hi = best_i * blk, min(n, (best_i + best_len) * blk)
    take = min(want, run_hi - run_lo)
    centre = 0.5 * (run_lo + run_hi)
    start = int(min(max(run_lo, centre - take / 2), run_hi - take))
    return start, start + take, best_len * blk / fs


def estimate(x_ct: np.ndarray, fs: int, n_rotors: int, cfg: Config) -> dict[str, Any]:
    """The full reading of one recording."""
    x = np.asarray(x_ct, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    n_total = x.shape[-1]
    start, stop, active_s = _stationary_window(x, fs, cfg)
    seg = x[:, start:stop]
    win_s = seg.shape[-1] / fs
    sp = make_spectrum(seg, fs, cfg)
    main = shaft_rate(sp, cfg)
    # Splitting is read on the DETECTED comb (where the line energy is) and
    # scaled back to shaft rev/s by the same factor the octave mapping applied.
    scale = main["rate_rev_s"] / max(main["detected_rev_s"], 1e-9)
    rotor = per_rotor(sp, main["detected_rev_s"], n_rotors, cfg)
    rotor = {
        **rotor,
        "speeds": [round(f * scale, 4) for f in rotor["speeds"]],
        "spread_rev_s": round(rotor["spread_rev_s"] * scale, 4),
    }

    # Two disjoint halves of the same window. A half is half as long, so its
    # odd-harmonic evidence is weaker and its own argmax may land on the
    # blade-pass double; that is an OCTAVE ambiguity, already decided on the
    # full window, not rate instability. The two half readings are therefore
    # octave-FOLDED before the +-1 rev/s stability rule is applied, and the
    # fold is recorded.
    half_n = seg.shape[-1] // 2
    halves: list[float] = []
    if half_n / fs >= cfg.min_window_s / 2.0:
        for a, b in ((0, half_n), (half_n, 2 * half_n)):
            sp_h = make_spectrum(seg[:, a:b], fs, cfg)
            halves.append(float(shaft_rate(sp_h, cfg)["rate_rev_s"]))
    folded = list(halves)
    half_fold = False
    if len(folded) == 2:
        ratio = folded[0] / max(folded[1], 1e-9)
        if abs(ratio - 2.0) <= 0.06:
            folded[0] /= 2.0
            half_fold = True
        elif abs(1.0 / ratio - 2.0) <= 0.06:
            folded[1] /= 2.0
            half_fold = True
        for i, value in enumerate(folded):  # also fold against the accepted rate
            if abs(value / max(main["rate_rev_s"], 1e-9) - 2.0) <= 0.06:
                folded[i] = value / 2.0
                half_fold = True
    half_delta = abs(folded[0] - folded[1]) if len(folded) == 2 else float("nan")

    df_rev_s = sp.df
    tol = max(
        0.0 if math.isnan(half_delta) else half_delta,
        0.5 * df_rev_s,
        0.25,
    )
    reasons: list[str] = []
    if win_s + 1e-6 < cfg.min_window_s:
        reasons.append(f"window {win_s:.2f} s < {cfg.min_window_s:.0f} s")
    if not (main["margin_db"] >= cfg.min_margin_db):
        reasons.append(
            f"harmonic-sum margin {main['margin_db']:.2f} dB < {cfg.min_margin_db:.0f} dB"
        )
    if math.isnan(half_delta):
        reasons.append("window too short for a two-half stability test")
    elif half_delta > cfg.max_half_delta:
        reasons.append(f"half-to-half {half_delta:.2f} rev/s > {cfg.max_half_delta:.1f}")
    if not (cfg.lo <= main["rate_rev_s"] <= cfg.hi):
        reasons.append(
            f"mapped shaft rate {main['rate_rev_s']:.2f} rev/s outside {cfg.lo:g}-{cfg.hi:g}"
        )

    speeds = rotor["speeds"] if rotor["n_distinct"] > 1 else [round(main["rate_rev_s"], 4)]
    return {
        "speed_rev_s": speeds,
        "rate_rev_s": round(float(main["rate_rev_s"]), 4),
        "n_distinct": rotor["n_distinct"],
        "spread_rev_s": rotor["spread_rev_s"],
        "spread_order": rotor["order"],
        "octave_verdict": main["octave_verdict"],
        "octave_trace": main["octave_trace"],
        "detected_rev_s": round(float(main["detected_rev_s"]), 4),
        "detected_score_db": round(float(main["detected_score_db"]), 3),
        "margin_db": round(float(main["margin_db"]), 3),
        "runner_up_rev_s": None
        if math.isnan(main["runner_up_rev_s"])
        else round(float(main["runner_up_rev_s"]), 3),
        "score_db": round(float(main["score_db"]), 3),
        "margin_octave_only_db": round(float(main["margin_octave_only_db"]), 3),
        "octave_only_runner_up_rev_s": None
        if math.isnan(main["octave_only_runner_up_rev_s"])
        else round(float(main["octave_only_runner_up_rev_s"]), 3),
        "half_rates_rev_s": [round(h, 4) for h in halves],
        "half_rates_folded_rev_s": [round(h, 4) for h in folded],
        "half_octave_folded": half_fold,
        "half_delta_rev_s": None if math.isnan(half_delta) else round(half_delta, 4),
        "speed_tolerance_rev_s": round(tol, 4),
        "window_s": round(win_s, 3),
        "window_start_s": round(start / fs, 3),
        "active_s": round(active_s, 3),
        "duration_s": round(n_total / fs, 3),
        "channels": int(x.shape[0]),
        "fs": int(fs),
        "df_hz": round(df_rev_s, 4),
        "usable": not reasons,
        "reason": "ok" if not reasons else "; ".join(reasons),
        "confidence": "high"
        if (not reasons and main["margin_db"] >= 6.0 and (half_delta or 0) <= 0.3)
        else ("medium" if not reasons else "low"),
    }


# ── Corpus readers: yield ``(meta dict, audio (C,T), fs)`` ───────────────────


def _mpath(frame: Any, path: str, default: Any = None) -> Any:
    """Dotted lookup into a recording Frame's nested ``meta`` sub-Frames."""
    if "meta" not in frame:
        return default
    node = frame["meta"]
    for part in path.split("."):
        if node is None or part not in node:
            return default
        node = node[part]
    return node


def _frames(dataset: str) -> Iterator[Any]:
    from data_processing.streams import iter_published_frames

    yield from iter_published_frames(dataset)


def _frame_audio(frame: Any) -> tuple[np.ndarray, int]:
    audio = frame["audio"]
    data = np.asarray(audio.data, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    return data, int(audio.tindex.sr)


def read_dregon_bench(limit: int | None) -> Iterator[tuple[dict, np.ndarray, int]]:
    seen = 0
    for frame in _frames("DREGON-frames"):
        rid = str(_mpath(frame, "recording_id", ""))
        if not rid.startswith("motor_"):
            continue
        m = re.match(r"motor_(Motor\d|allMotors)_(\d+)$", rid)
        if m is None:
            continue
        motor, throttle = m.group(1), int(m.group(2))
        single = motor != "allMotors"
        audio, fs = _frame_audio(frame)
        yield (
            {
                "corpus": "DREGON-bench",
                "id": rid,
                "rig": "dregon_mikrokopter" + ("_single_motor" if single else "_quad"),
                "condition": f"bench {motor} throttle {throttle}%",
                "throttle": float(throttle),
                "n_rotors": N_ROTORS["dregon_bench_single" if single else "dregon_bench_all"],
                "family": "dregon_bench",
            },
            audio,
            fs,
        )
        seen += 1
        if limit and seen >= limit:
            return


_SPCUP_CONDITIONS = ("single_rotor", "stationary", "hover")


def read_spcup(limit: int | None) -> Iterator[tuple[dict, np.ndarray, int]]:
    seen = 0
    for frame in _frames("SPCUP19-egonoise"):
        cond = _mpath(frame, "operating.condition")
        if cond not in _SPCUP_CONDITIONS:
            continue
        rid = str(_mpath(frame, "recording_id", ""))
        team = str(_mpath(frame, "system.team", "?"))
        model = str(_mpath(frame, "system.make_model", "?"))
        audio, fs = _frame_audio(frame)
        single = cond == "single_rotor"
        yield (
            {
                "corpus": "SPCUP19-egonoise",
                "id": rid,
                "rig": f"spcup_{team}",
                "rig_model": model,
                "condition": cond,
                "throttle": None,
                "n_rotors": N_ROTORS["spcup_single_rotor" if single else "spcup_quad"],
                "family": "spcup_single" if single else "spcup_static",
            },
            audio,
            fs,
        )
        seen += 1
        if limit and seen >= limit:
            return


def read_kaist(limit: int | None) -> Iterator[tuple[dict, np.ndarray, int]]:
    for seen, frame in enumerate(_frames("KAIST-rotating-acoustic"), start=1):
        rid = str(_mpath(frame, "recording_id", ""))
        audio, fs = _frame_audio(frame)
        yield (
            {
                "corpus": "KAIST-rotating-acoustic",
                "id": rid,
                "rig": "kaist_rotating_testbed",
                "condition": "bench 3010 RPM, 0 Nm",
                "throttle": None,
                "n_rotors": N_ROTORS["kaist"],
                "family": "kaist",
            },
            audio,
            fs,
        )
        if limit and seen >= limit:
            return


def daset_cells(paths: list[str]) -> dict[str, dict[str, str]]:
    """``file_path`` → the design cell of a DroneAudioSet drone-only recording."""
    out: dict[str, dict[str, str]] = {}
    for fp in paths:
        low = fp.lower()
        drone = "drone2" if "drone2" in low else "drone1"
        thr = "high" if "throttle-high" in low else "low"
        dist = "25cm" if "25cm" in low else "50cm"
        mic = "M_down" if "8array-down" in low else ("M_up" if "8array-up" in low else "M_center")
        out[fp] = {"drone": drone, "throttle": thr, "mic_dist": dist, "mic": mic}
    return out


def read_daset(limit: int | None, daset_dir: Path) -> Iterator[tuple[dict, np.ndarray, int]]:
    import pyarrow.parquet as pq

    from data_processing.sources.droneaudio import _arrow_list_scalar_to_ct

    files = sorted(Path(daset_dir).glob("*.parquet"))
    if not files:
        raise SystemExit(
            f"no parquet under {daset_dir} — fetch the drone-only subset first "
            "(scripts/noise_v2_bench_speed.py --fetch-daset)"
        )
    seen = 0
    for path in files:
        pf = pq.ParquetFile(str(path))
        for batch in pf.iter_batches(batch_size=2):
            col = batch.column("audio")
            arrays = col.field("array")
            srs = col.field("sampling_rate").to_pylist()
            fps = batch.column("file_path").to_pylist()
            for i in range(batch.num_rows):
                fp = str(fps[i])
                cell = daset_cells([fp])[fp]
                audio = _arrow_list_scalar_to_ct(arrays[i])
                take = Path(fp).stem.split("-")[-1]
                yield (
                    {
                        "corpus": "DroneAudioSet",
                        "id": f"{cell['drone']}_{cell['throttle']}_{cell['mic_dist']}_"
                        f"{cell['mic']}_{take}",
                        "rig": f"daset_{cell['drone']}",
                        "condition": f"rig-mounted static, throttle {cell['throttle']}, "
                        f"mic {cell['mic']} at {cell['mic_dist']}",
                        "throttle": None,
                        "throttle_level": cell["throttle"],
                        "mic": cell["mic"],
                        "mic_dist": cell["mic_dist"],
                        "drone": cell["drone"],
                        "raw_path": fp,
                        "n_rotors": N_ROTORS["daset"],
                        "family": "daset",
                    },
                    audio,
                    int(srs[i]),
                )
                seen += 1
                if limit and seen >= limit:
                    return


def read_drone_audio(limit: int | None, root: Path) -> Iterator[tuple[dict, np.ndarray, int]]:
    import soundfile as sf

    files = sorted((root / "Binary_Drone_Audio" / "yes_drone").glob("*.wav"))
    if not files:
        raise SystemExit(f"no yes_drone wavs under {root}")
    # one clip per take prefix, in order, so the sample spans the takes
    by_prefix: dict[str, list[Path]] = {}
    for p in files:
        by_prefix.setdefault(re.sub(r"_\d+_\.wav$", "", p.name), []).append(p)
    seen = 0
    for prefix, clips in sorted(by_prefix.items()):
        for clip in clips[:1]:
            data, fs = sf.read(str(clip), dtype="float32", always_2d=True)
            drone = "bebop" if "bebop" in prefix else ("membo" if "membo" in prefix else "unknown")
            yield (
                {
                    "corpus": "drone_audio",
                    "id": prefix,
                    "rig": f"alemadi_{drone}",
                    "condition": "indoor propeller recording, 1 s clip",
                    "throttle": None,
                    "n_rotors": N_ROTORS["drone_audio"],
                    "family": "drone_audio",
                },
                np.ascontiguousarray(data.T),
                int(fs),
            )
            seen += 1
            if limit and seen >= limit:
                return


def read_zenodo(limit: int | None, root: Path) -> Iterator[tuple[dict, np.ndarray, int]]:
    import soundfile as sf

    files = sorted((root / "raw").rglob("*.wav"))
    if not files:
        raise SystemExit(f"no wavs under {root}/raw")
    for seen, path in enumerate(files, start=1):
        data, fs = sf.read(str(path), dtype="float32", always_2d=True)
        yield (
            {
                "corpus": "zenodo_drone_noises",
                "id": path.stem,
                "rig": f"zenodo_unknown_{path.stem}",
                "condition": f"unlabelled drone-noise clip ({path.parent.name})",
                "throttle": None,
                "n_rotors": N_ROTORS["zenodo"],
                "family": "zenodo",
            },
            np.ascontiguousarray(data.T),
            int(fs),
        )
        if limit and seen >= limit:
            return


def read_chums(limit: int | None, mat_path: Path) -> Iterator[tuple[dict, np.ndarray, int]]:
    """SPCUP19 ChuMS ``UAV_rotor_recordings.mat`` — a propeller test rig.

    The file holds one ``TestResults`` struct: ``MicPositions`` (8x2, mm) and
    ``Test(1..9)``, each ``Test`` a run described by ``Details`` ("N
    propellers. Repeat:R") whose ``Data(1..8)`` are the EIGHT MICROPHONES of
    that run (``RawTruncatedCalibrated``, Pa, 44.1 kHz, 73-75 s, plus a
    per-mic ``OASPL`` and a pre-computed ``Freq``/``SPL`` spectrum). The mics
    are truncated to slightly different lengths, so they are stacked to the
    common minimum. No RPM or throttle is recorded anywhere in the file.
    """
    from scipy.io import loadmat

    if not Path(mat_path).exists():
        raise SystemExit(
            f"{mat_path} not found — fetch it first (scripts/noise_v2_bench_speed.py --fetch-chums)"
        )
    mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    root = np.atleast_1d(mat["TestResults"])[0]
    mic_pos = np.asarray(getattr(root, "MicPositions", np.zeros((0, 2)))).tolist()
    for ti, test in enumerate(np.atleast_1d(root.Test)):
        details = str(getattr(test, "Details", f"test {ti}"))
        mics = np.atleast_1d(getattr(test, "Data", []))
        sigs = [
            np.asarray(d.RawTruncatedCalibrated, dtype=np.float32).reshape(-1)
            for d in mics
            if getattr(d, "RawTruncatedCalibrated", None) is not None
        ]
        if not sigs:
            continue
        n = min(s.size for s in sigs)
        audio = np.stack([s[:n] for s in sigs], axis=0)
        fs = int(round(float(getattr(mics[0], "Fs", 44100))))
        prop_match = re.match(r"(\d+)", details)
        n_props = int(prop_match.group(1)) if prop_match else 1
        repeat = int(m.group(1)) if (m := re.search(r"Repeat:(\d+)", details)) else 1
        yield (
            {
                "corpus": "SPCUP19-ChuMS-bench",
                "id": f"{n_props}prop_repeat{repeat}",
                "rig": f"spcup_ChuMS_{n_props}prop_bench",
                "rig_model": "Skylark M4-680 (Dotterel) propeller rig",
                "condition": details,
                "throttle": None,
                "n_props": n_props,
                "repeat": repeat,
                "oaspl_db": [round(float(getattr(d, "OASPL", float("nan"))), 2) for d in mics],
                "mic_positions_mm": mic_pos,
                "n_rotors": n_props,
                "family": "chums_bench",
            },
            audio,
            fs,
        )
        if limit and ti + 1 >= limit:
            return


def _jsonable(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.dtype.kind in "US":
        return arr.tolist()
    if arr.dtype.kind in "fiub":
        return arr.tolist()
    return str(value)


READERS = {
    "dregon_bench": lambda a: read_dregon_bench(a.limit),
    "spcup": lambda a: read_spcup(a.limit),
    "daset": lambda a: read_daset(a.limit, a.daset_dir),
    "drone_audio": lambda a: read_drone_audio(a.limit, a.drone_audio_dir),
    "zenodo": lambda a: read_zenodo(a.limit, a.zenodo_dir),
    "kaist": lambda a: read_kaist(a.limit),
    "chums": lambda a: read_chums(a.limit, a.chums_mat),
}


# ── Cross-checks ─────────────────────────────────────────────────────────────


def cross_checks(rows: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    slope, icept = DREGON_LAW

    bench = [r for r in rows if r["family"] == "dregon_bench" and r["n_rotors"] == 1]
    if bench:
        err = [r["rate_rev_s"] - (slope * r["throttle"] + icept) for r in bench]
        thr = np.array([r["throttle"] for r in bench], dtype=float)
        est = np.array([r["rate_rev_s"] for r in bench], dtype=float)
        fit = np.polyfit(thr, est, 1)
        pred = np.polyval(fit, thr)
        ss_res = float(((est - pred) ** 2).sum())
        ss_tot = float(((est - est.mean()) ** 2).sum())
        out["dregon_throttle_law"] = {
            "n": len(bench),
            "law": {"slope": slope, "intercept": icept},
            "mean_abs_error_rev_s": round(float(np.mean(np.abs(err))), 4),
            "max_abs_error_rev_s": round(float(np.max(np.abs(err))), 4),
            "rms_error_rev_s": round(float(np.sqrt(np.mean(np.square(err)))), 4),
            "refit": {
                "slope": round(float(fit[0]), 5),
                "intercept": round(float(fit[1]), 4),
                "r2": round(1.0 - ss_res / ss_tot, 6) if ss_tot > 0 else None,
                "residual_rms_rev_s": round(float(np.sqrt(ss_res / len(bench))), 4),
            },
            "per_throttle_mean": {
                str(int(t)): round(float(est[thr == t].mean()), 3)
                for t in sorted(set(thr.tolist()))
            },
            "octave_verdicts": {
                v: sum(1 for r in bench if r["octave_verdict"] == v)
                for v in sorted({r["octave_verdict"] for r in bench})
            },
        }

    agh = {
        r["id"].rsplit("__", 1)[-1]: r
        for r in rows
        if r["corpus"] == "SPCUP19-egonoise" and "single_rotors" in r["id"]
    }
    pairs = [(SPCUP_AGH_BLIND[k], agh[k]["rate_rev_s"]) for k in SPCUP_AGH_BLIND if k in agh]
    if pairs:
        d = np.array([b - a for a, b in pairs])
        out["spcup_agh_blind"] = {
            "n": len(pairs),
            "mean_abs_error_rev_s": round(float(np.mean(np.abs(d))), 4),
            "max_abs_error_rev_s": round(float(np.max(np.abs(d))), 4),
            "per_take": {
                k: {
                    "blind_rev_s": SPCUP_AGH_BLIND[k],
                    "welch_rev_s": agh[k]["rate_rev_s"],
                    "delta_rev_s": round(agh[k]["rate_rev_s"] - SPCUP_AGH_BLIND[k], 3),
                    "octave_verdict": agh[k]["octave_verdict"],
                }
                for k in sorted(SPCUP_AGH_BLIND)
                if k in agh
            },
        }

    kaist = [r for r in rows if r["family"] == "kaist"]
    if kaist:
        out["kaist_nominal"] = {
            "nominal_rev_s": round(KAIST_NOMINAL_REV_S, 4),
            "per_recording": {
                r["id"]: {
                    "rev_s": r["rate_rev_s"],
                    "rel_error_pct": round(100.0 * (r["rate_rev_s"] / KAIST_NOMINAL_REV_S - 1), 3),
                    "usable": r["usable"],
                }
                for r in kaist
            },
        }

    daset = [r for r in rows if r["family"] == "daset" and r["usable"]]
    if daset:
        cells: dict[str, list[float]] = {}
        for r in daset:
            cells.setdefault(f"{r['drone']}|{r['throttle_level']}", []).append(r["rate_rev_s"])
        out["daset_cells"] = {
            k: {
                "n": len(v),
                "mean_rev_s": round(float(np.mean(v)), 3),
                "std_rev_s": round(float(np.std(v)), 3),
                "blade_pass_hz": round(2.0 * float(np.mean(v)), 2),
            }
            for k, v in sorted(cells.items())
        }
        out["daset_paper_lines_hz"] = {f"{k[0]}|{k[1]}": v for k, v in DASET_PAPER_LINE_HZ.items()}
    return out


# ── Figures ──────────────────────────────────────────────────────────────────


def draw_figures(rows: list[dict], checks: dict, out_dir: Path, fig_dirs: list[Path]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    names: list[str] = []
    bench = sorted(
        (r for r in rows if r["family"] == "dregon_bench" and r["n_rotors"] == 1),
        key=lambda r: r["throttle"],
    )
    if bench:
        fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=150)
        thr = np.array([r["throttle"] for r in bench])
        est = np.array([r["rate_rev_s"] for r in bench])
        slope, icept = DREGON_LAW
        grid = np.linspace(thr.min() - 3, thr.max() + 3, 50)
        ax.plot(
            grid, slope * grid + icept, "k--", lw=1.2, label=f"blind law {slope}·throttle + {icept}"
        )
        ax.plot(thr, est, "o", ms=6, color="#1f77b4", label="Welch harmonic-sum estimate")
        for r in bench:
            if r["octave_verdict"] != "as_found":
                ax.plot(
                    [r["throttle"]],
                    [r["rate_rev_s"]],
                    "o",
                    ms=11,
                    mfc="none",
                    mec="#d62728",
                    mew=1.4,
                )
        ax.set_xlabel("throttle setpoint (%)")
        ax.set_ylabel("shaft rate (rev/s)")
        law = checks.get("dregon_throttle_law", {})
        ax.set_title(
            "DREGON single-motor bench: estimate vs the validated throttle law\n"
            f"mean |error| {law.get('mean_abs_error_rev_s', float('nan')):.2f} rev/s, "
            f"refit R² {law.get('refit', {}).get('r2', float('nan')):.5f}"
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        names.append(_save(fig, "bench_speeds_dregon_law.png", out_dir, fig_dirs))
        plt.close(fig)

    # margin vs stability scatter, all corpora
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    colors = {}
    for r in rows:
        c = colors.setdefault(r["corpus"], f"C{len(colors)}")
        x = r["margin_db"] if np.isfinite(r["margin_db"]) else 25.0
        y = r["half_delta_rev_s"]
        y = 10.0 if y is None else max(y, 0.01)
        ax.scatter(x, y, s=22, color=c, alpha=0.75, marker="o" if r["usable"] else "x")
    ax.axvline(3.0, color="k", ls="--", lw=1.0)
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlabel("harmonic-sum margin over the non-octave runner-up (dB)")
    ax.set_ylabel("half-to-half rate difference (rev/s, log)")
    ax.set_title("Tolerance rule: margin ≥ 3 dB and half-to-half ≤ 1 rev/s\n(o usable, x rejected)")
    handles = [Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in colors.items()]
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    names.append(_save(fig, "bench_speeds_gates.png", out_dir, fig_dirs))
    plt.close(fig)
    return names


def _save(fig: Any, name: str, out_dir: Path, fig_dirs: list[Path]) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name)
    for d in fig_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name)
    return name


# ── Fetch helper (DroneAudioSet drone-only subset) ───────────────────────────


def fetch_daset(dest: Path) -> None:
    """Download the 28 ``drone-only`` parquet shards (3.1 GiB) from HuggingFace."""
    import requests

    dest.mkdir(parents=True, exist_ok=True)
    base = (
        "https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet/"
        "resolve/main/drone-only/train_%03d-00000-of-00001.parquet"
    )
    session = requests.Session()
    for i in range(1, 29):
        path = dest / f"train_{i:03d}.parquet"
        url = base % i
        head = session.head(url, allow_redirects=True, timeout=120)
        want = int(head.headers.get("content-length", 0))
        if path.exists() and (want == 0 or path.stat().st_size == want):
            continue
        print(f"  fetching {path.name} ({want / 2**20:.0f} MiB)", flush=True)
        with session.get(url, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            with open(path, "wb") as fh:
                for chunk in resp.iter_content(1 << 22):
                    fh.write(chunk)


CHUMS_ZIP_URL = "https://dregon.inria.fr/SPCup2019_egonoise/SPCUP19_ChuMS_data.zip"


def fetch_chums(mat_path: Path) -> None:
    """Download + extract the ChuMS propeller-rig ``.mat`` (1.36 GiB zip).

    The INRIA host stalls on a single streamed GET but serves byte ranges
    fine, so the zip is pulled in 64 MiB range chunks and then extracted.
    """
    import zipfile

    import requests

    mat_path = Path(mat_path)
    if mat_path.exists():
        return
    mat_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = mat_path.parent / "SPCUP19_ChuMS_data.zip"
    session = requests.Session()
    total = int(
        session.head(CHUMS_ZIP_URL, allow_redirects=True, timeout=120).headers["content-length"]
    )
    if not zip_path.exists() or zip_path.stat().st_size != total:
        chunk = 64 << 20
        with open(zip_path, "wb") as fh:
            for start in range(0, total, chunk):
                end = min(start + chunk, total) - 1
                resp = session.get(
                    CHUMS_ZIP_URL, headers={"Range": f"bytes={start}-{end}"}, timeout=600
                )
                resp.raise_for_status()
                fh.write(resp.content)
                print(f"  chums {(end + 1) / total:6.1%}", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("UAV_rotor_recordings.mat", mat_path.parent)
    zip_path.unlink(missing_ok=True)


# ── Main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--corpora",
        default="dregon_bench",
        help="comma list of " + ",".join(READERS) + ", or 'all'",
    )
    ap.add_argument("--limit", type=int, default=0, help="stop after N recordings per corpus")
    ap.add_argument("--out", type=Path, default=_ROOT / "results/noise_v2/survey")
    ap.add_argument(
        "--fig-dir",
        type=Path,
        default=_ROOT / "docs/explainers/noise-model-v2-plan",
        help="second copy of every figure",
    )
    ap.add_argument("--daset-dir", type=Path, default=_ROOT / ".cache/source_raw/noise_v2/daset")
    ap.add_argument("--drone-audio-dir", type=Path, default=_ROOT / "data/drone_audio")
    ap.add_argument("--zenodo-dir", type=Path, default=_ROOT / "data/zenodo_drone_noises")
    ap.add_argument(
        "--chums-mat",
        type=Path,
        default=_ROOT / ".cache/source_raw/noise_v2/chums/UAV_rotor_recordings.mat",
    )
    ap.add_argument("--fetch-daset", action="store_true", help="download the drone-only shards")
    ap.add_argument("--fetch-chums", action="store_true", help="download the ChuMS rotor-rig mat")
    ap.add_argument("--window-s", type=float, default=Config.window_s)
    ap.add_argument("--min-window-s", type=float, default=Config.min_window_s)
    ap.add_argument("--min-margin-db", type=float, default=Config.min_margin_db)
    ap.add_argument("--max-half-delta", type=float, default=Config.max_half_delta)
    ap.add_argument("--odd-margin-db", type=float, default=Config.odd_margin_db)
    ap.add_argument("--active-range-db", type=float, default=Config.active_range_db)
    ap.add_argument("--level-tol-db", type=float, default=Config.level_tol_db)
    ap.add_argument("--lo", type=float, default=Config.lo)
    ap.add_argument("--hi", type=float, default=Config.hi)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    args = ap.parse_args(argv)

    if args.fetch_daset:
        fetch_daset(args.daset_dir)

    if args.fetch_chums:
        fetch_chums(args.chums_mat)
    cfg = Config(
        lo=args.lo,
        hi=args.hi,
        window_s=args.window_s,
        min_window_s=args.min_window_s,
        min_margin_db=args.min_margin_db,
        max_half_delta=args.max_half_delta,
        odd_margin_db=args.odd_margin_db,
        active_range_db=args.active_range_db,
        level_tol_db=args.level_tol_db,
    )
    wanted = list(READERS) if args.corpora == "all" else args.corpora.split(",")
    unknown = [w for w in wanted if w not in READERS]
    if unknown:
        ap.error(f"unknown corpora {unknown}; known: {list(READERS)}")

    limit = args.limit or None
    args.limit = limit
    rows: list[dict] = []
    failures: list[dict] = []
    for name in wanted:
        print(f"[{name}] reading ...", flush=True)
        try:
            stream = READERS[name](args)
        except SystemExit as exc:  # a missing raw tree is a survey fact, not a crash
            failures.append({"corpus": name, "error": str(exc)})
            print(f"[{name}] SKIPPED: {exc}", flush=True)
            continue
        n = 0
        for meta, audio, fs in stream:
            reading = estimate(audio, fs, int(meta["n_rotors"]), cfg)
            row = {**meta, **reading}
            rows.append(row)
            n += 1
            print(
                f"  {row['corpus']}/{row['id']}: {row['speed_rev_s']} rev/s "
                f"({row['octave_verdict']}, margin {row['margin_db']} dB, "
                f"usable={row['usable']})",
                flush=True,
            )
        print(f"[{name}] {n} recordings", flush=True)

    checks = cross_checks(rows)
    figures = [] if args.no_figures else draw_figures(rows, checks, args.out, [args.fig_dir])

    payload = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()},
        "tolerance_rule": {
            "min_margin_db": cfg.min_margin_db,
            "max_half_delta_rev_s": cfg.max_half_delta,
            "min_window_s": cfg.min_window_s,
            "statement": (
                "usable iff the harmonic-sum peak clears the best non-octave runner-up by "
                f">= {cfg.min_margin_db:g} dB AND the two disjoint halves of the window agree "
                f"to <= {cfg.max_half_delta:g} rev/s AND the stationary window is "
                f">= {cfg.min_window_s:g} s"
            ),
        },
        "corpora": wanted,
        "n_recordings": len(rows),
        "n_usable": sum(1 for r in rows if r["usable"]),
        "skipped_corpora": failures,
        "cross_checks": checks,
        "figures": figures,
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    path = args.out / f"bench_speeds{tag}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str))
    print(f"\nwrote {path} — {len(rows)} recordings, {payload['n_usable']} usable")
    if checks.get("dregon_throttle_law"):
        law = checks["dregon_throttle_law"]
        print(
            f"DREGON law cross-check: mean |error| {law['mean_abs_error_rev_s']} rev/s "
            f"(max {law['max_abs_error_rev_s']}), refit slope {law['refit']['slope']} "
            f"intercept {law['refit']['intercept']} R2 {law['refit']['r2']}"
        )
    if checks.get("spcup_agh_blind"):
        agh = checks["spcup_agh_blind"]
        print(
            f"SPCUP AGH cross-check: mean |error| {agh['mean_abs_error_rev_s']} rev/s "
            f"(max {agh['max_abs_error_rev_s']}, n={agh['n']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
