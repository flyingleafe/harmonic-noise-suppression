"""Microphone wind / downwash buffeting from PCM, graded by evidence.

What is measured
----------------
Wind and rotor downwash on a microphone are pseudo-sound: turbulent pressure on the
capsule. It is read on a 4 kHz copy of each channel in 128 ms Hann frames (64 ms hop,
7.8 Hz bins; frame mean = DC bias removed), per channel and per 2 s block:

1. **Line-free LF floor.** Periodic lines (rotor/engine harmonics, mains, beating rotor
   clusters) are masked per block: bins > 3 dB over a 9-bin running median of the
   block-mean log spectrum, +-1 bin. ``tilt`` = line-free 23-188 Hz floor over the
   line-free 500-1500 Hz floor; three octave sub-bands (23-47, 47-94, 94-188 Hz) test
   that the excess is broadband, not one resonance.
2. **Comodulation** ``rho``: buffeting raises the whole LF band at once. Mean
   correlation, across non-adjacent line-free LF bins, of their 128 ms level changes
   (null 0 +- ``NULL_RHO_SD`` per block for any stationary floor incl. self-noise;
   slow fly-by/throttle trends cancel). ``rho_ref`` is the same in 500-1500 Hz: speech,
   thumps and handling comodulate both bands.
3. **Spatial incoherence** (multichannel roles): turbulence decorrelates within
   centimetres, acoustic LF (wavelength 1.8-15 m) is coherent across a compact array.
   Median LF magnitude-squared coherence (MSC) of each channel with its partners
   (pairs <= ``MAX_PAIR_M`` apart when positions are known); ``coherence_ref`` = best
   MSC among the OTHER channels' pairs, i.e. whether the array holds an acoustic
   reference; ``disparity`` = LF floor over the array's calm (25th pct) channel.

Evidence modes and block decisions (``classify_block``): with an acoustic reference
(``coherence_ref >= COH_ACOUSTIC``) a channel is wind when its incoherent LF power is
>= ``EXCESS_WIND_DB`` over its coherent part, MSC <= ``COH_WIND``, LF-dominant,
comodulated and broadband; clean when its incoherent part does not exceed the coherent
one. A compact array of known geometry without ANY coherent pair (``incoherent_array``,
e.g. a flying onboard array) is wind only with strong LF-specific comodulation (self-noise
is stationary), LF dominance, broadband excess and an LF floor above ``LEVEL_INCOH_DB``.
Without spatial evidence (mono, unknown geometry, wide arrays) buffeting cannot be
separated from handling/structure-borne LF: such blocks are counted
(``buffeting_like_s``) but never certified; only a block without LF dominance is clean.
A high-passed LF (``lf_attenuated``) is never called clean on tilt alone.

Significance (operational, per channel): ``significant`` when wind blocks cover >=
``SIG_FRACTION`` of the assessed time; ``mixed`` (intermittent) when >= ``MIXED_FRACTION``
and >= ``MIXED_MIN_S`` s; ``not_significant`` only when wind < ``MIXED_FRACTION`` and
clean >= ``CLEAN_FRACTION``; else ``unknown``. A role takes its worst channel (one windy
capsule contaminates the role) with the split in ``windy_channels``/``channels``.
Setting (indoor/outdoor), weather-station wind and documented labels are never inputs;
documented source evidence wins over PCM in :func:`combine`.

Input scale and cost: float PCM at full scale (+-1), any rate >= 3.1 kHz; scale
invariant except digital silence/padding (``DEAD_DBFS``, ``SILENT_POWER``: excluded
blocks), clipping (reported) and ``LEVEL_INCOH_DB``. numpy/scipy only; one polyphase
resample and a 512-point rfft per 64 ms (~1.6 ms per mono 1 s clip, ~1.5 s per 60 s
8-ch 44.1 kHz view). Views < ``MIN_VIEW_S`` are refused (``too_short``).

Validation (uni-cpu jobs wind-val-a/b/c, block features of every view of 11+3 parents)
-------------------------------------------------------------------------------------
Thresholds were set on the DREGON single-motor takes (20 takes x 8 ch; documented
downwash capsules Motor1 ch5/6, Motor2 ch0, Motor3 ch1/2, Motor4 ch4 from
docs/experiments/static-rig-profiles.md). Across TILT_MIN 8-15 dB, COH_WIND 0.15-0.3,
RHO_MIN 0.03-0.1 and EXCESS_WIND 2-6 dB all 25 documented capsules stay flagged and the
flagged neighbour count moves between 1 and 7 of 130; dropping the LF-level disparity
(the measure behind those labels) changes no call.
In-sample: 25/25 documented capsules flagged (24 significant, 1 mixed) and 5/5 of the
uncertain Motor1 ch5; of 130 other channel-takes 109 not_significant, 15 unknown, 6
flagged - all on symmetric neighbours (Motor4 ch3 x3, Motor3 ch0 x2, Motor2 ch7) that
are LF-incoherent with 11-14 dB LF excess, i.e. partially exposed, not labelled.
Held out: AuDroK anechoic Mi4 (ch3) whose FF/FB transits carry a documented 30-60 dB
LF gain: 4/8 FF/FB flagged on ch3 (3 significant, 1 mixed), 4 unknown (no acoustic
reference), 0 not_significant; the 8 H1/H2 hovers flag nothing. Documented no-airflow
or rotor-off indoor controls: DREGON clean loudspeaker takes 198 not_significant / 42
unknown / 0 flagged of 240; AGH calibration + static-clean 11/1/0 of 12; KumamoTech
rotor-off tail, SDU noise floor not_significant; ANU room (incoherent self-noise)
unknown; Salford coupler calibrations (mono) 9 unknown - their LF is buffeting-like,
which is why mono is never certified. UaVirBASE weather wind (0-2.9 m/s, auxiliary)
does not correlate with buffeting-like time (Spearman 0.03): too narrow a range to test.
Disagreement kept: the SPCUP19 KumamoTech annotator's in-flight ch12 "+13-14 dB at
100-400 Hz (downwash/turbulence)" is LF-coherent with the array (excess -1.8 dB):
PCM says not_significant. Not validated: the incoherent_array mode (no documented
positive/negative beyond AGH 'static corrupted' "+32/35 dB rotor/downwash 60-400 Hz",
consistent; it flags all 8 DREGON flights and allMotors_70), mono files, arrays with unknown geometry
and no coherent pair (UaVirBASE mast, AuDroK outdoor).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from fractions import Fraction
from typing import Any

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import resample_poly

RECIPE = "wind/1"

#: Views shorter than this are not assessed (``too_short``); a 1 s view is one block.
MIN_VIEW_S = 1.0
#: Runner guidance: view length for long takes, the longest single view, view cap.
PREFERRED_VIEW_S = 30.0
MAX_VIEW_S = 60.0
MAX_VIEWS = 8
#: At most this many channels of one role are assessed (evenly chosen).
MAX_CHANNELS = 16

FS = 4000
NFFT = 512
HOP = 256
BIN_HZ = FS / NFFT
LF = (3, 25)  # 23.4-195 Hz bin slice (bins 3..24 -> 23-188 Hz centres)
SUBBANDS = ((3, 6), (6, 12), (12, 25))  # 23-47, 47-94, 94-188 Hz
REF = (64, 193)  # 500-1500 Hz
LINE_WINDOW_BINS = 9  # 70 Hz running median of the block-mean log spectrum
LINE_DB = 3.0  # a bin this far above it (or its neighbour) is a periodic line
BLOCK_S = 2.0
FRAME_S = NFFT / FS
HOP_S = HOP / FS

DEAD_DBFS = -110.0
SILENT_POWER = 1e-14  # mean-square per 4 kHz frame: exact zeros / padding (-140 dBFS)
CLIP_LEVEL = 0.999
SOUND_SPEED = 343.0
MAX_PAIR_M = 0.5
#: Diffuse-field MSC at 100 Hz for MAX_PAIR_M is ~0.75; pairs farther apart are not used.

#: Stationary Gaussian floor: median ``gust_db`` of this estimator (dB) and the per-block
#: SD of ``rho`` (synthetic white/pink noise, tests/data_processing/test_wind_annotation.py).
NULL_GUST_DB = 1.3
NULL_RHO_SD = 0.035

#: Calibrated on uni-cpu (jobs wind-val-a/b/c, 2026-11-27) against the documented DREGON
#: single-motor downwash capsules and rotor-off/indoor/calibrator controls; see the
#: module docstring "Validation". Changing any value changes ``RECIPE``.
THRESHOLDS: dict[str, float] = {
    # --- acoustic-reference mode: the rest of the array is LF-coherent (acoustic) ---
    "COH_ACOUSTIC": 0.5,  # max MSC among the other channels' pairs that makes them a reference
    "COH_WIND": 0.2,  # the channel's own median MSC with its partners at most this
    "EXCESS_WIND_DB": 4.0,  # incoherent/coherent LF power, 10 log10(coh_ref/coh - 1)
    "EXCESS_CLEAN_DB": 0.0,  # at most this: incoherent part <= acoustic part -> clean
    "TILT_MIN_DB": 10.0,  # line-free 23-188 Hz floor over the 500-1500 Hz floor
    "RHO_MIN": 0.05,  # LF comodulation (null 0 +- 0.035 per block)
    # --- incoherent-array mode: compact known geometry, no LF coherence anywhere ---
    "RHO_INCOH": 0.12,
    "TILT_INCOH_DB": 12.0,
    "LEVEL_INCOH_DB": -40.0,  # LF floor density (dB re full scale) above self-noise levels
    # --- broadband: >= MIN_SUBBANDS of the 3 LF octaves exceed the 500-1500 Hz floor by
    #     SUB_EXCESS_DB (or, with a reference array, the calm channel by DISPARITY_MIN_DB) ---
    "MIN_SUBBANDS": 2,
    "SUB_EXCESS_DB": 8.0,
    "DISPARITY_MIN_DB": 6.0,
    "RHO_MARGIN": 0.05,  # LF comodulation must exceed the 500-1500 Hz one (speech, thumps)
    # --- clean without spatial evidence: no LF dominance at all ---
    "TILT_CLEAN_DB": 8.0,
    "DISP_CLEAN_DB": 6.0,
    # --- mono / no spatial evidence: buffeting-like blocks are counted, never certified ---
    "RHO_MONO": 0.15,
    "TILT_MONO_DB": 20.0,
    # high-pass: 94-188 Hz this far over 23-47 Hz and 23-47 Hz under the 500-1500 Hz floor
    "HP_DROP_DB": 15.0,
    # --- time aggregation per channel ---
    "SIG_FRACTION": 0.3,
    "MIXED_FRACTION": 0.05,
    "MIXED_MIN_S": 2.0,
    "CLEAN_FRACTION": 0.7,
}

_SEVERITY = {"significant": 3, "mixed": 2, "not_significant": 1, "unknown": 0, "not_applicable": -1}
_HANN = np.hanning(NFFT + 1)[:-1].astype(np.float32)
_MAX_KEEP = REF[1]


def too_short(duration_s: float, role: str = "audio") -> dict:
    """The result for a view too short to assess (callers may skip decoding it)."""
    return _empty(
        role,
        "too_short",
        f"{duration_s:.2f} s < {MIN_VIEW_S} s minimum view",
        duration_s=duration_s,
    )


def _empty(role: str, reason: str, detail: str, **extra: Any) -> dict:
    return {
        "wind": "unknown",
        "reason": reason,
        "detail": detail,
        "role": role,
        "confidence": 0.0,
        "coverage_s": 0.0,
        "wind_s": 0.0,
        "clean_s": 0.0,
        "ambiguous_s": 0.0,
        "buffeting_like_s": 0.0,
        "spatial": "unavailable",
        "channels": [],
        "recipe": RECIPE,
        **extra,
    }


def _to_4k(x: np.ndarray, sr: float) -> np.ndarray:
    if sr == FS:
        return x
    ratio = Fraction(FS) / Fraction(sr).limit_denominator(1_000_000)
    ratio = ratio.limit_denominator(4096)
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=-1).astype(
        np.float32, copy=False
    )


def _pair_ok(positions: Any, n: int) -> np.ndarray | None:
    """(n, n) bool: pairs expected to be acoustically coherent at LF; None if unknown."""
    if positions is None:
        return None
    try:
        p = np.asarray(positions, dtype=float)
    except (TypeError, ValueError):
        return None
    if p.ndim != 2 or p.shape[0] != n or p.shape[1] < 2:
        return None
    d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    ok = np.isfinite(d) & (d <= MAX_PAIR_M) & (d > 0.0)
    if not np.isfinite(d[~np.eye(n, dtype=bool)]).any():
        return None
    return ok


def _pick_channels(n: int, wanted: Any) -> list[int]:
    chans = sorted({int(c) for c in wanted if 0 <= int(c) < n}) if wanted else list(range(n))
    if len(chans) > MAX_CHANNELS:
        idx = np.linspace(0, len(chans) - 1, MAX_CHANNELS).round().astype(int)
        chans = [chans[i] for i in idx]
    return chans


def _db(p: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(p, 1e-30))


def _nanmedian(a: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def _comodulation(level_db: np.ndarray, free: np.ndarray) -> np.ndarray:
    """(C, T, F) frame log-power, (C, F) usable bins -> (C,) mean correlation, across
    non-adjacent bins, of their 128 ms level changes (frames t+2 vs t, disjoint pairs).

    Buffeting modulates the whole LF band at once (positive); a stationary floor gives
    0 +- 0.035 per 2 s block (independent chi-square bins; adjacent Hann bins excluded);
    slow level trends (fly-by, throttle) cancel in the differences."""
    d = (level_db[:, 2:] - level_db[:, :-2])[:, ::2]
    d = d - d.mean(axis=1, keepdims=True)
    sd = d.std(axis=1, keepdims=True)
    z = np.where((sd > 0) & free[:, None, :], d / np.where(sd > 0, sd, 1.0), 0.0)
    total = (z.sum(-1) ** 2).sum(1)
    diag = (z**2).sum((1, 2))
    adjacent = (z[:, :, 1:] * z[:, :, :-1]).sum((1, 2))
    n = free.sum(-1)
    pairs = n * n - n - 2 * (free[:, 1:] & free[:, :-1]).sum(-1)
    out = (total - diag - 2 * adjacent) / np.maximum(pairs, 1) / z.shape[1]
    return np.where(pairs >= 6, out, np.nan)


def _robust_sd(d: np.ndarray, axis: int = -1) -> np.ndarray:
    med = np.median(d, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(d - med), axis=axis)


def _line_mask(pm: np.ndarray) -> np.ndarray:
    """(C, F) block-mean power -> True where a periodic line (or its Hann neighbour) sits.

    The running median of the log spectrum preserves monotonic slopes (steep wind
    spectra stay unmasked) and ignores up to 4 line bins in the 9-bin window (close
    rotor clusters, mains, engine harmonics)."""
    lp = _db(pm[:, 2:])
    trend = median_filter(lp, size=(1, LINE_WINDOW_BINS), mode="nearest")
    peak = lp - trend > LINE_DB
    peak[:, 1:] |= peak[:, :-1].copy()
    peak[:, :-1] |= peak[:, 1:].copy()
    mask = np.zeros(pm.shape, dtype=bool)
    mask[:, 2:] = peak
    return mask


def _band_free(mask: np.ndarray, a: int, b: int, pm: np.ndarray) -> np.ndarray:
    """(C, b-a) weights of line-free bins; when lines cover > 75 % of the band, its
    lowest-quartile bins (the floor between dense lines) instead."""
    free = ~mask[:, a:b]
    few = free.sum(axis=1) < max(1, (b - a) // 4)
    if few.any():
        q = np.percentile(pm[few, a:b], 25, axis=1, keepdims=True)
        free[few] = pm[few, a:b] <= q
    return free.astype(np.float32)


def block_features(audio_ct: Any, sr: float, facts: Mapping[str, Any] | None = None) -> dict | str:
    """Per-``BLOCK_S`` block, per-channel features of one view (what ``assess`` decides on).

    Returns ``'too_short'`` / ``'sample_rate_too_low'`` or a dict with ``channels``,
    ``duration_s``, per-channel ``dead``/``level_dbfs``/``dc``/``clip``/``hp_drop_db``/
    ``lf_attenuated``, ``pair_ok`` and ``blocks`` (list of dicts of (C,) arrays)."""
    facts = facts or {}
    x = np.asarray(audio_ct, dtype=np.float32)
    if x.ndim == 1:
        x = x[None]
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError(f"audio_ct must be (channels, samples), got {x.shape}")
    sr = float(sr)
    duration = x.shape[1] / sr
    if duration < MIN_VIEW_S:
        return "too_short"
    if sr < 2 * REF[1] * BIN_HZ:
        return "sample_rate_too_low"
    chans = _pick_channels(x.shape[0], facts.get("wind_channels"))
    x = x[chans]
    ok = np.isfinite(x)
    xz = np.where(ok, x, 0.0).astype(np.float32, copy=False)
    n_ok = np.maximum(ok.sum(axis=1), 1)
    dc = xz.sum(axis=1) / n_ok
    level_dbfs = _db((xz.astype(np.float64) ** 2).sum(axis=1) / n_ok)
    clip = (np.abs(xz) >= CLIP_LEVEL).sum(axis=1) / n_ok
    y = _to_4k(xz - dc[:, None].astype(np.float32), sr)
    n_frames = 1 + (y.shape[1] - NFFT) // HOP
    if n_frames < 8:
        return "too_short"
    frames = np.lib.stride_tricks.sliding_window_view(y, NFFT, axis=1)[:, ::HOP][:, :n_frames]
    frames = frames - frames.mean(axis=2, keepdims=True)
    spec = np.fft.rfft(frames * _HANN, axis=2)[:, :, :_MAX_KEEP]  # (C, T, F)
    power = spec.real**2 + spec.imag**2
    silent = (frames**2).mean(axis=2) <= SILENT_POWER  # zero padding, replaced non-finite spans
    n_ch = len(chans)
    dead = level_dbfs < DEAD_DBFS
    pos = facts.get("mic_positions")
    if pos is not None:
        try:
            pos = [pos[c] for c in chans]
        except (IndexError, KeyError, TypeError):
            pos = None
    pair_ok = _pair_ok(pos, n_ch)

    per_block = max(1, int(round(BLOCK_S / HOP_S)))
    starts = list(range(0, n_frames, per_block))
    if len(starts) > 1 and n_frames - starts[-1] < per_block // 2:
        starts.pop()
    bounds = [(a, starts[i + 1] if i + 1 < len(starts) else n_frames) for i, a in enumerate(starts)]
    blocks = []
    for a, b in bounds:
        span_s = (b - a) * HOP_S if b < n_frames else (b - a - 1) * HOP_S + FRAME_S
        live = ~dead & ~silent[:, a:b].any(axis=1)
        p = power[:, a:b]
        pm = p.mean(axis=1)  # (C, F)
        mask = _line_mask(pm)
        w_lf = _band_free(mask, *LF, pm)
        w_ref = _band_free(mask, *REF, pm)
        e_lf = _db(
            (p[:, :, LF[0] : LF[1]] * w_lf[:, None]).sum(-1) / w_lf.sum(-1)[:, None]
        )  # (C, T)
        e_ref = _db((p[:, :, REF[0] : REF[1]] * w_ref[:, None]).sum(-1) / w_ref.sum(-1)[:, None])
        lf_blk = _db((pm[:, LF[0] : LF[1]] * w_lf).sum(-1) / w_lf.sum(-1))
        ref_blk = _db((pm[:, REF[0] : REF[1]] * w_ref).sum(-1) / w_ref.sum(-1))
        tonal = _db(pm[:, LF[0] : LF[1]].mean(-1)) - lf_blk
        sub = np.full((3, n_ch), np.nan)
        for i, (sa, sb) in enumerate(SUBBANDS):
            w = w_lf[:, sa - LF[0] : sb - LF[0]]
            has = w.sum(-1) > 0
            sub[i, has] = _db((pm[has, sa:sb] * w[has]).sum(-1) / w[has].sum(-1))
        d_lf = (e_lf[:, 2:] - e_lf[:, :-2])[:, ::2]
        d_ref = (e_ref[:, 2:] - e_ref[:, :-2])[:, ::2]
        if d_lf.shape[1] >= 4:
            gust = _robust_sd(d_lf) / np.sqrt(2.0)
            gust_ref = _robust_sd(d_ref) / np.sqrt(2.0)
            rho = _comodulation(_db(p[:, :, LF[0] : LF[1]]), w_lf > 0)
            rho_ref = _comodulation(_db(p[:, :, REF[0] : REF[1]]), w_ref > 0)
        else:
            gust = gust_ref = rho = rho_ref = np.full(n_ch, np.nan)
        coh = np.full(n_ch, np.nan)
        coh_ref = np.full(n_ch, np.nan)
        disparity = np.full(n_ch, np.nan)
        sub_disp = np.full((3, n_ch), np.nan)
        if live.sum() >= 2:
            z = spec[:, a:b, LF[0] : LF[1]]
            sxx = p[:, :, LF[0] : LF[1]].sum(axis=1)  # (C, F)
            sxy = np.einsum("itf,jtf->ijf", z, z.conj())
            msc = np.abs(sxy) ** 2 / np.maximum(sxx[:, None] * sxx[None, :], 1e-30)
            both = (w_lf[:, None] > 0) & (w_lf[None, :] > 0)
            msc = np.where(both, msc, np.nan)
            msc_med = _nanmedian(msc, axis=2)
            np.fill_diagonal(msc_med, np.nan)
            msc_med[~live] = np.nan
            msc_med[:, ~live] = np.nan
            if pair_ok is not None:
                msc_med = np.where(pair_ok, msc_med, np.nan)
            for c in range(n_ch):
                if np.isfinite(msc_med[c]).any():
                    coh[c] = float(np.nanmedian(msc_med[c]))
                others = np.delete(np.delete(msc_med, c, 0), c, 1)
                if np.isfinite(others).any():
                    coh_ref[c] = float(np.nanmax(others))
            calm = np.percentile(lf_blk[live], 25) if live.sum() >= 3 else lf_blk[live].min()
            disparity = np.where(live, lf_blk - calm, np.nan)
            for i in range(3):
                s = sub[i]
                okc = live & np.isfinite(s)
                if okc.sum() >= 2:
                    ref = np.percentile(s[okc], 25) if okc.sum() >= 3 else s[okc].min()
                    sub_disp[i] = np.where(okc, s - ref, np.nan)
        sub_x = sub - ref_blk[None]
        sub_x[:, ~live] = np.nan
        blocks.append(
            {
                "start_s": a * HOP_S,
                "span_s": span_s,
                "lf_db": lf_blk,
                "ref_db": ref_blk,
                "tilt_db": lf_blk - ref_blk,
                "sub_excess_db": sub_x,
                "sub_disparity_db": sub_disp,
                "gust_db": gust,
                "gust_ref_db": gust_ref,
                "rho": rho,
                "rho_ref": rho_ref,
                "tonal_db": tonal,
                "lf_free_bins": w_lf.sum(-1),
                "coherence": coh,
                "coherence_ref": coh_ref,
                "disparity_db": disparity,
                "pair_geometry": pair_ok is not None,
                "valid": live,
            }
        )
    sub_all = np.stack([blk["sub_excess_db"] for blk in blocks])  # (B, 3, C)
    hp_drop = _nanmedian(sub_all[:, 2] - sub_all[:, 0], axis=0)
    lf_attenuated = (np.nan_to_num(hp_drop, nan=0.0) >= THRESHOLDS["HP_DROP_DB"]) & (
        np.nan_to_num(_nanmedian(sub_all[:, 0], axis=0), nan=0.0) < 0.0
    )
    return {
        "channels": [int(c) for c in chans],
        "duration_s": duration,
        "dead": dead,
        "level_dbfs": level_dbfs,
        "dc": dc,
        "clip": clip,
        "hp_drop_db": hp_drop,
        "lf_attenuated": lf_attenuated,
        "pair_geometry": pair_ok is not None,
        "blocks": blocks,
    }


def _n_subbands(
    blk: Mapping[str, Any], c: int, th: Mapping[str, float], *, with_disparity: bool
) -> int:
    sx = np.asarray(blk["sub_excess_db"], dtype=float)[:, c]
    hit = np.nan_to_num(sx, nan=-np.inf) >= th["SUB_EXCESS_DB"]
    if with_disparity:
        sd = np.asarray(blk["sub_disparity_db"], dtype=float)[:, c]
        hit |= np.nan_to_num(sd, nan=-np.inf) >= th["DISPARITY_MIN_DB"]
    return int(hit.sum())


def classify_block(
    blk: Mapping[str, Any],
    dead: np.ndarray,
    lf_attenuated: np.ndarray,
    th: Mapping[str, float] = THRESHOLDS,
) -> tuple[list[str], list[str], list[bool]]:
    """Per channel of one block: state (``wind``/``clean``/``ambiguous``), spatial evidence
    mode (``acoustic_reference``/``incoherent_array``/``unavailable``) and whether the block
    is buffeting-like without spatial evidence (counted, never certified).

    acoustic_reference: the other channels hold an LF-coherent (acoustic) pair; wind = the
    channel is incoherent with them (MSC <= COH_WIND and incoherent/coherent LF power >=
    EXCESS_WIND_DB), LF-dominant, comodulated and broadband; clean = its incoherent part
    does not exceed the coherent one. incoherent_array: known compact geometry (pairs <=
    MAX_PAIR_M, where acoustic LF must be coherent) but no coherent pair at all; wind needs
    stronger comodulation than the 500-1500 Hz band (self-noise is stationary), LF
    dominance, broadband and an LF floor above LEVEL_INCOH_DB. Otherwise only a block
    without LF dominance is clean."""
    states, modes, buffet = [], [], []
    for c in range(len(dead)):
        if dead[c] or not blk["valid"][c]:
            states.append("ambiguous")
            modes.append("unavailable")
            buffet.append(False)
            continue
        coh, coh_ref = float(blk["coherence"][c]), float(blk["coherence_ref"][c])
        tilt, rho, rho_ref = (
            float(blk["tilt_db"][c]),
            float(blk["rho"][c]),
            float(blk["rho_ref"][c]),
        )
        disp = float(np.nan_to_num(blk["disparity_db"][c], nan=0.0))
        lf_specific = rho >= rho_ref + th["RHO_MARGIN"]
        mode = "unavailable"
        if np.isfinite(coh) and coh_ref >= th["COH_ACOUSTIC"]:
            mode = "acoustic_reference"
        elif np.isfinite(coh) and blk["pair_geometry"] and coh_ref < th["COH_WIND"]:
            mode = "incoherent_array"
        wind = clean = False
        if mode == "acoustic_reference":
            excess = 10.0 * np.log10(max(coh_ref / max(coh, 1e-6) - 1.0, 1e-3))
            wind = (
                excess >= th["EXCESS_WIND_DB"]
                and coh <= th["COH_WIND"]
                and tilt >= th["TILT_MIN_DB"]
                and rho >= th["RHO_MIN"]
                and _n_subbands(blk, c, th, with_disparity=True) >= th["MIN_SUBBANDS"]
            )
            clean = excess <= th["EXCESS_CLEAN_DB"]
        elif mode == "incoherent_array":
            wind = (
                rho >= th["RHO_INCOH"]
                and lf_specific
                and tilt >= th["TILT_INCOH_DB"]
                and float(blk["lf_db"][c]) >= th["LEVEL_INCOH_DB"]
                and _n_subbands(blk, c, th, with_disparity=False) >= th["MIN_SUBBANDS"]
            )
        clean = clean or (
            tilt < th["TILT_CLEAN_DB"] and disp < th["DISP_CLEAN_DB"] and not lf_attenuated[c]
        )
        buffet.append(
            bool(
                mode != "acoustic_reference"
                and not wind
                and rho >= th["RHO_MONO"]
                and lf_specific
                and tilt >= th["TILT_MONO_DB"]
                and _n_subbands(blk, c, th, with_disparity=False) >= th["MIN_SUBBANDS"]
            )
        )
        states.append("wind" if wind else "clean" if clean else "ambiguous")
        modes.append(mode)
    return states, modes, buffet


def assess(
    audio_ct: Any, sr: float, facts: Mapping[str, Any] | None = None, *, role: str = "audio"
) -> dict:
    """Wind/downwash contamination of ONE view of one audio role.

    ``audio_ct``: (channels, samples) or (samples,) float PCM at full scale.
    ``facts`` (all optional): ``mic_positions`` (channels x 3 m, NaN/None allowed),
    ``wind_channels`` (subset to assess), ``environment`` (``simulated`` ->
    not_applicable). Returns a JSON-native dict: ``wind``, ``reason``, ``detail``,
    ``confidence``, ``coverage_s``/``wind_s``/``clean_s``/``ambiguous_s``/
    ``buffeting_like_s`` (of the worst channel), ``spatial`` (evidence mode behind the
    verdict), ``windy_channels``/``calm_channels`` and per-channel ``channels`` entries
    (verdict, seconds per state and per spatial mode, block-median metrics), ``recipe``.
    """
    facts = facts or {}
    if facts.get("environment") == "simulated":
        return {
            **_empty(role, "simulated", "synthetic auralisation: no microphone airflow"),
            "wind": "not_applicable",
        }
    feat = block_features(audio_ct, sr, facts)
    if isinstance(feat, str):
        x = np.asarray(audio_ct)
        duration = x.shape[-1] / float(sr)
        if feat == "too_short":
            return too_short(duration, role)
        return _empty(role, feat, f"{float(sr):g} Hz cannot cover the 1.5 kHz reference band")
    n_ch = len(feat["channels"])
    acc = {c: {"rows": [], **{k: 0.0 for k in _SECONDS}} for c in range(n_ch)}
    dead, lf_attenuated = feat["dead"], feat["lf_attenuated"]
    for blk in feat["blocks"]:
        states, modes, buffet = classify_block(blk, dead, lf_attenuated)
        for c in range(n_ch):
            if dead[c] or not blk["valid"][c]:
                continue
            span = blk["span_s"]
            acc[c][f"{states[c]}_s"] += span
            acc[c][f"{modes[c]}_s"] += span
            if states[c] == "wind":
                acc[c][f"wind_{modes[c]}_s"] += span
            if buffet[c]:
                acc[c]["buffeting_like_s"] += span
            acc[c]["rows"].append(
                (
                    span,
                    blk["lf_db"][c],
                    blk["tilt_db"][c],
                    blk["gust_db"][c],
                    blk["gust_ref_db"][c],
                    blk["tonal_db"][c],
                    blk["coherence"][c],
                    blk["disparity_db"][c],
                    blk["coherence_ref"][c],
                    blk["rho"][c],
                    blk["rho_ref"][c],
                )
            )
    chans, level_dbfs, dc, clip, hp_drop = (
        feat[k] for k in ("channels", "level_dbfs", "dc", "clip", "hp_drop_db")
    )
    channels = []
    for c in range(n_ch):
        rows = acc[c]["rows"]
        entry: dict[str, Any] = {
            "channel": int(chans[c]),
            "dead": bool(dead[c]),
            "level_dbfs": _r(level_dbfs[c]),
            "dc_offset": _r(dc[c], 6),
            "clip_fraction": _r(clip[c], 6),
            "lf_attenuated": bool(lf_attenuated[c]),
            "hp_drop_db": _r(hp_drop[c]),
        }
        entry.update({k: _r(v, 3) for k, v in acc[c].items() if k.endswith("_s")})
        if rows:
            r = np.array(rows, dtype=float)
            w = r[:, 0]
            entry.update(
                covered_s=_r(w.sum(), 3),
                lf_db=_wmed(r[:, 1], w),
                tilt_db=_wmed(r[:, 2], w),
                tilt_p90_db=_p90(r[:, 2]),
                gust_db=_wmed(r[:, 3], w),
                gust_ref_db=_wmed(r[:, 4], w),
                tonal_db=_wmed(r[:, 5], w),
                coherence=_wmed(r[:, 6], w),
                disparity_db=_wmed(r[:, 7], w),
                disparity_p90_db=_p90(r[:, 7]),
                coherence_ref=_wmed(r[:, 8], w),
                rho=_wmed(r[:, 9], w),
                rho_p90=_p90(r[:, 9]),
                rho_ref=_wmed(r[:, 10], w),
            )
        else:
            entry["covered_s"] = 0.0
        entry["wind"] = _label(entry)
        channels.append(entry)
    return _role_result(channels, role=role, duration_s=feat["duration_s"])


def _r(v: Any, nd: int = 2) -> float | None:
    v = float(v)
    return round(v, nd) if np.isfinite(v) else None


def _p90(v: np.ndarray) -> float | None:
    v = v[np.isfinite(v)]
    return _r(np.percentile(v, 90)) if v.size else None


def _wmed(v: np.ndarray, w: np.ndarray) -> float | None:
    ok = np.isfinite(v)
    if not ok.any():
        return None
    v, w = v[ok], w[ok]
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return _r(v[o][np.searchsorted(cw, cw[-1] / 2)])


def _label(ch: Mapping[str, Any]) -> str:
    th = THRESHOLDS
    cov = float(ch.get("covered_s") or 0.0)
    if cov <= 0 or ch.get("dead"):
        return "unknown"
    fw = ch["wind_s"] / cov
    if fw >= th["SIG_FRACTION"]:
        return "significant"
    if fw >= th["MIXED_FRACTION"] and ch["wind_s"] >= min(th["MIXED_MIN_S"], cov):
        return "mixed"
    if fw < th["MIXED_FRACTION"] and ch["clean_s"] / cov >= th["CLEAN_FRACTION"]:
        return "not_significant"
    return "unknown"


#: Evidence weight of a verdict by the spatial mode that carried it (see module docstring).
_CONFIDENCE = {"acoustic_reference": 0.85, "incoherent_array": 0.6, "unavailable": 0.7}


def _role_result(channels: list[dict], *, role: str, **extra: Any) -> dict:
    live = [c for c in channels if not c.get("dead") and (c.get("covered_s") or 0) > 0]
    if not live:
        return _empty(
            role,
            "no_live_channel",
            "every assessed channel is digital silence, padding or non-finite",
            channels=channels,
            **extra,
        )
    labels = [c["wind"] for c in live]
    worst = max(live, key=lambda c: (_SEVERITY[c["wind"]], (c["wind_s"] or 0) / c["covered_s"]))
    if any(lab in ("significant", "mixed") for lab in labels):
        wind = worst["wind"]
    elif all(lab == "not_significant" for lab in labels):
        wind = "not_significant"
    else:
        wind = "unknown"
    windy = [c["channel"] for c in live if c["wind"] in ("significant", "mixed")]
    calm = [c["channel"] for c in live if c["wind"] == "not_significant"]
    cov = worst["covered_s"]
    if wind in ("significant", "mixed"):
        spatial = max(
            ("acoustic_reference", "incoherent_array"),
            key=lambda m: worst.get(f"wind_{m}_s") or 0.0,
        )
        frac = worst["wind_s"] / cov
        conf = _CONFIDENCE[spatial] * min(1.0, frac / THRESHOLDS["SIG_FRACTION"]) ** 0.5
        reason = "buffeting" if spatial == "acoustic_reference" else "buffeting_incoherent_array"
        detail = (
            f"incoherent unsteady broadband LF on ch {windy} of {len(live)} ({spatial}): "
            f"{worst['wind_s']:.1f}/{cov:.1f} s, tilt {worst.get('tilt_db')} dB, coherence "
            f"{worst.get('coherence')} vs array {worst.get('coherence_ref')}, comodulation {worst.get('rho')}"
        )
    elif wind == "not_significant":
        ref = sum(c.get("acoustic_reference_s") or 0.0 for c in live) / sum(
            c["covered_s"] for c in live
        )
        spatial = "acoustic_reference" if ref >= 0.5 else "unavailable"
        clean = min(c["clean_s"] / c["covered_s"] for c in live)
        conf = _CONFIDENCE[spatial] * clean
        reason = "no_buffeting"
        detail = (
            f"{len(live)} ch LF-coherent with the array or without LF dominance for >= {clean:.0%} "
            f"of {cov:.1f} s"
        )
    else:
        spatial = max(_MODES, key=lambda m: sum(c.get(f"{m}_s") or 0.0 for c in live))
        conf = 0.0
        amb = max(c["ambiguous_s"] / c["covered_s"] for c in live)
        buff = max((c.get("buffeting_like_s") or 0.0) / c["covered_s"] for c in live)
        if buff >= THRESHOLDS["SIG_FRACTION"]:
            reason = "buffeting_like_unverified"
            detail = (
                f"unsteady broadband LF for {buff:.0%} of the time but no spatial evidence "
                "(mono, unknown geometry or wide array): wind, handling and structure-borne LF not separable"
            )
        elif any(c["lf_attenuated"] for c in live):
            reason = "lf_attenuated"
            detail = f"{amb:.0%} of the time neither buffeting nor clean; LF high-passed"
        else:
            reason = "ambiguous"
            detail = f"{amb:.0%} of the time LF-dominant without decisive coherence/comodulation"
    if cov < 5.0:
        conf *= 0.7
    return {
        "wind": wind,
        "reason": reason,
        "detail": detail,
        "role": role,
        "confidence": round(conf, 3),
        "coverage_s": _r(cov, 3),
        "wind_s": _r(worst["wind_s"], 3),
        "clean_s": _r(worst["clean_s"], 3),
        "ambiguous_s": _r(worst["ambiguous_s"], 3),
        "buffeting_like_s": _r(worst.get("buffeting_like_s") or 0.0, 3),
        "spatial": spatial,
        "windy_channels": windy,
        "calm_channels": calm,
        "channels": channels,
        "recipe": RECIPE,
        **extra,
    }


_STATES = ("wind", "clean", "ambiguous")
_MODES = ("acoustic_reference", "incoherent_array", "unavailable")
_SECONDS = (
    *(f"{s}_s" for s in _STATES),
    "buffeting_like_s",
    *(f"{m}_s" for m in _MODES),
    *(f"wind_{m}_s" for m in _MODES),
)
_WEIGHTED = (
    "lf_db",
    "tilt_db",
    "gust_db",
    "gust_ref_db",
    "tonal_db",
    "coherence",
    "coherence_ref",
    "disparity_db",
    "level_dbfs",
    "rho",
    "rho_ref",
)
_MAXED = ("tilt_p90_db", "disparity_p90_db", "rho_p90", "clip_fraction", "hp_drop_db")


def merge(results: Iterable[Mapping[str, Any]], *, role: str | None = None) -> dict:
    """Combine ``assess`` results of several views of the SAME role (e.g. evenly spaced
    30 s views of a long take): per-channel seconds are summed, metrics coverage-weighted,
    labels recomputed by the same rule. Views without coverage are ignored."""
    results = list(results)
    parts = [r for r in results if r.get("channels") and (r.get("coverage_s") or 0) > 0]
    role = role or str(next((r["role"] for r in results if r.get("role")), "audio"))
    if not parts:
        return _empty(role, "too_short", "no view could be assessed")
    by_ch: dict[int, list[Mapping]] = {}
    for r in parts:
        for ch in r["channels"]:
            by_ch.setdefault(int(ch["channel"]), []).append(ch)
    channels = []
    for c, items in sorted(by_ch.items()):
        live = [i for i in items if not i.get("dead") and (i.get("covered_s") or 0) > 0]
        entry: dict[str, Any] = {"channel": c, "dead": not live, "n_views": len(items)}
        entry["lf_attenuated"] = any(i.get("lf_attenuated") for i in live)
        for k in ("covered_s", *_SECONDS):
            entry[k] = _r(sum(float(i.get(k) or 0.0) for i in live), 3)
        w = np.array([float(i["covered_s"]) for i in live])
        for k in _WEIGHTED:
            v = np.array([np.nan if i.get(k) is None else float(i[k]) for i in live])
            ok = np.isfinite(v)
            entry[k] = _r((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else None
        for k in _MAXED:
            v = [float(i[k]) for i in live if i.get(k) is not None]
            entry[k] = max(v) if v else None
        entry["wind"] = _label(entry) if live else "unknown"
        channels.append(entry)
    return _role_result(
        channels, role=role, n_views=len(parts), view_labels=[r["wind"] for r in parts]
    )


def combine(
    role_results: Mapping[str, Mapping[str, Any]],
    direct: Mapping[str, Any] | None = None,
    *,
    wind_role: str | None = None,
) -> dict:
    """Recording-level ``wind`` + ``role_wind`` + evidence from per-role PCM results.

    ``direct`` = the classifier's documented facet (``{'wind': value, 'evidence': text}``,
    optionally ``role_wind``). A documented definite value wins over PCM; PCM
    disagreement is kept in ``evidence`` and ``wind_metrics``. ``wind_role``
    (classifier's role of interest) decides the recording value when given and
    assessed; otherwise significant if every assessed role is significant, mixed if
    roles disagree with a windy one, not_significant only if every role is
    not_significant."""
    direct = dict(direct or {})
    role_wind = {r: res["wind"] for r, res in role_results.items()}
    for r, v in (direct.get("role_wind") or {}).items():
        if v not in ("unknown", None):
            role_wind[r] = v
    metrics = {
        r: {
            k: res.get(k)
            for k in (
                "reason",
                "confidence",
                "coverage_s",
                "wind_s",
                "clean_s",
                "ambiguous_s",
                "buffeting_like_s",
                "spatial",
                "windy_channels",
                "calm_channels",
            )
        }
        for r, res in role_results.items()
    }
    if wind_role and role_wind.get(wind_role) not in (None, "unknown"):
        pcm = role_wind[wind_role]
    else:
        definite = [v for v in role_wind.values() if v not in ("unknown", "not_applicable")]
        if not definite:
            pcm = "unknown"
        elif all(v == "significant" for v in definite) and len(definite) == len(role_wind):
            pcm = "significant"
        elif any(v in ("significant", "mixed") for v in definite):
            pcm = (
                "mixed" if len(set(definite)) > 1 or len(definite) < len(role_wind) else definite[0]
            )
        elif all(v == "not_significant" for v in role_wind.values()):
            pcm = "not_significant"
        else:
            pcm = "unknown"
    parts = []
    for r, res in role_results.items():
        parts.append(
            f"{r}: {res['wind']} ({res.get('reason')}, conf {res.get('confidence')}; {res.get('detail', '')})"
        )
    pcm_text = f"PCM {RECIPE}: " + "; ".join(parts) if parts else ""
    dv = direct.get("wind")
    if dv not in (None, "unknown"):
        wind = dv
        text = str(direct.get("evidence") or "documented by source")
        if pcm not in ("unknown", dv) and parts:
            text += f" [PCM disagrees: {pcm_text}]"
    else:
        wind = pcm
        text = pcm_text or "no PCM view assessed"
    return {
        "wind": wind,
        "role_wind": role_wind,
        "evidence": text,
        "wind_metrics": metrics,
        "recipe": RECIPE,
    }
