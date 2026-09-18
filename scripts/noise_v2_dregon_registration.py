#!/usr/bin/env python
"""Do the DREGON telemetry labels REGISTER the real comb lines on the five score windows?

The R3 plan (``docs/explainers/noise-model-v2-plan.qmd``, ``#model-r3``) freezes
the DREGON comb from the bench and fits one comb level ``c`` on the five room-2
cruise score windows. R2 fitted ``c = -3.2`` dB while the band-level match that
the HPPNet gate likes needs ``+21.8`` dB (``round2/render_dregon/findings.md``
lines 196-200): a ~25 dB disagreement between the Whittle likelihood and the
gate about how loud the comb is. Either reading is possible only if the label
carrier lands on the real lines. This script decides that, read-only.

Per window (5) x label track (``rps_refined``, ``motors_command``) x rotor (4):

(a) the label's mean carrier and its within-window / within-frame drift;
(b) the registration score ``S(delta) = sum_k max P`` over a window of
    ``2 bins + 0.3 k drift`` around ``k (f_r + delta)`` of the mic-mean
    periodogram, ``delta`` on a 0.002 rev/s grid over +-1.5 rev/s: argmax
    ``delta*``, and ``S`` at ``delta = 0`` against ``S(delta*)``;
(c) per-order line SNR (peak over the local-median floor, dB) at the label and
    at ``delta*``, with the count of orders above 6 dB;
(d) the separability restriction: with four rotors 2.2-3.0 rev/s apart only the
    orders with ``k dF > 2 bins`` carry a per-rotor line, and (b)/(c) are
    summed over those alone. Orders whose search window would exceed half the
    line spacing because the carrier DRIFTS inside the analysis frame are
    reported unresolvable too.
(e) the same statistics on three analysis lengths: the whole 4 s window
    (0.25 Hz bins), 1.024 s sub-frames and the flight likelihood's own
    ``2048``-sample / hop-512 periodic-Hann frames (7.81 Hz bins), so the
    number says what the Whittle objective actually sees.

Plus one ORDER-TRACKED cross-check the assignment does not name but the drift
forces: these "cruise" windows sweep 3.5-25 rev/s inside 4 s, so a plain 4 s
periodogram smears every line over tens of bins and cannot answer (b) at all.
Resampling each window onto the label's own uniform-revolution grid turns every
order of one rotor into a constant-frequency line at once, so the full 4 s of
coherence is available at 1/(f T) ~ 0.003 orders and a label bias of 0.01 rev/s
is visible at high k. It is the best case for the label: if no comb clears the
floor THERE, the comb really is quiet.

Reuse: the score / line-SNR statistics are those of
``scripts/noise_v2_bench_diag.py`` (``half_width_bins``, ``score_curve``,
``line_table``) and the per-frame label-tracked line reading is that of
``scripts/noise_v2_render_dregon.py`` (``tracked_line``, ``encoded_speed_dev``).
They are re-expressed here rather than imported because both modules import
``experiments.noise_model.spectrum``/``model``, which R3 is rewriting
concurrently; this script imports only ``gates``, ``supports`` and ``clips``.

Outputs (``results/noise_v2/rounds/round3/registration/``): one JSON per
window, ``registration.json`` with the roll-up, three PNGs and ``findings.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy.ndimage import maximum_filter1d, median_filter  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from experiments.noise_model import gates as GT  # noqa: E402
from experiments.noise_model import supports as SU  # noqa: E402
from experiments.stochastic_fit import clips as C  # noqa: E402

SCHEMA = "noise-v2-dregon-registration/1"
OUT_DIR = Path("results/noise_v2/rounds/round3/registration")
#: The R2 frozen-comb flight fit whose ``comb_gain_db`` this check is about.
R2_FIT = Path("results/noise_v2/rounds/round2/fits/dregon_room2_floor__flight_floor_only_v2.json")
#: The band-level-matched comb shift of ``round2/render_dregon/findings.md:196``.
LEVEL_MATCHED_SHIFT_DB = 21.80

#: Registration grid: +-1.5 rev/s around the label carrier, 0.002 rev/s.
REG_HALF_WIDTH = 1.5
REG_STEP = 0.002
#: Search window of one order: ``2 bins + 0.3 k drift`` (drift in rev/s, so
#: ``k drift`` is the Hz the line sweeps inside the analysis frame).
WINDOW_BINS = 2.0
DRIFT_FACTOR = 0.3
#: An order is only usable if its search window stays inside half the line
#: spacing ``f_r / 2``; beyond that the window swallows the neighbouring order.
WINDOW_SPACING_FRACTION = 0.5
#: A line must clear its local floor by this to count (the bench-diag threshold).
LINE_SNR_THRESHOLD_DB = 6.0
#: Cap on one order's contribution to the registration score, in dB: four fixed
#: room interferers would otherwise decide a sum over ~80 rotor orders.
SNR_CAP_DB = 10.0
#: Local floor: median over +-30 Hz, never fewer than 8 bins a side.
FLOOR_HALF_HZ = 30.0
FLOOR_MIN_BINS = 8
#: Order tracking: upsample before the non-uniform interpolation (linear interp
#: on a 16 kHz signal with content at 8 kHz would cost several dB), and allow
#: this much intrinsic line width when reading a resampled line.
ORDER_TRACK_UP = 4
ORDER_TRACK_WIDTH_HZ = 4.0

#: The two DREGON rotor tracks: the refined posterior label the R3 spec calls
#: "the label", and the raw command track the R2 fits actually ran on.
LABEL_KEYS = ("rps_refined", "motors_command")

BAND_LO = SU.BAND_LEVEL_F_MIN
BAND_HI = SU.BAND_LEVEL_F_MAX


@dataclass(frozen=True)
class Analysis:
    """One analysis length: ``hop=None`` is the single whole-window frame."""

    name: str
    n: int
    hop: int | None

    def starts(self, n_samples: int) -> np.ndarray:
        n = min(self.n, n_samples)
        if self.hop is None:
            return np.zeros(1, dtype=np.int64)
        return np.arange(0, n_samples - n + 1, self.hop, dtype=np.int64)


ANALYSES: tuple[Analysis, ...] = (
    Analysis("window_4s", 64000, None),
    Analysis("frame_1.024s", 16384, 8192),
    Analysis(f"frame_{SU.OBS_N_FFT}", SU.OBS_N_FFT, SU.OBS_HOP),
)
#: The analysis the verdict rests on: the flight likelihood's own front end.
VERDICT_ANALYSIS = f"frame_{SU.OBS_N_FFT}"


def die(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - guard
        return "unknown"


def db(v: Any) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(v, dtype=np.float64), 1e-300))


def _r(v: Any, nd: int = 3) -> Any:
    if v is None:
        return None
    x = float(v)
    return round(x, nd) if math.isfinite(x) else None


# ── the window: audio at 16 kHz plus the label on the audio grid ────────────


def load_window(support: GT.ScoredSupport, rps_key: str) -> tuple[np.ndarray, np.ndarray]:
    """``(audio (M, T), label (R, T))`` at 16 kHz — the support the fit sees."""
    clip = C.load_clip(
        SU.DREGON_DATASET,
        support.recording,
        support.start_s,
        support.duration_s,
        version=None,
        channels=None,
        rps_key=rps_key,
        clip_id=f"reg_{support.recording}@{support.start_s:.3f}_{rps_key}",
    )
    clip = C.decimate(clip, SU.SR)
    if clip.rps is None:
        die(f"{support.recording}: {rps_key} carries no rotor track")
    return np.asarray(clip.audio, dtype=np.float64), np.asarray(clip.rps, dtype=np.float64)


def frame_power(audio: np.ndarray, starts: np.ndarray, n: int) -> np.ndarray:
    """Mic-MEAN periodogram ``(n_frames, F)``, periodic Hann, the front end's scale."""
    w = np.hanning(n + 1)[:n]
    seg = audio[:, starts[:, None] + np.arange(n)[None, :]] * w
    power = np.abs(np.fft.rfft(seg, axis=-1)) ** 2 / float((w**2).sum())
    return np.asarray(power.mean(axis=0), dtype=np.float64)


def frame_labels(label: np.ndarray, starts: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``(mean (R, n_frames), drift (R, n_frames))`` of the label inside each frame."""
    seg = label[:, starts[:, None] + np.arange(n)[None, :]]
    return seg.mean(axis=-1), seg.max(axis=-1) - seg.min(axis=-1)


def local_floor(pm: np.ndarray, df: float) -> np.ndarray:
    half = max(FLOOR_MIN_BINS, int(round(FLOOR_HALF_HZ / df)))
    return np.asarray(median_filter(pm, size=(1, 2 * half + 1), mode="nearest"))


# ── order geometry: which orders carry a per-rotor line at all ──────────────


def half_width_bins(k: int, df: float, drift_rev_s: float) -> int:
    """``2 bins + 0.3 k drift`` in bins — ``bench_diag.half_width_bins(kind="spec")``
    with the within-frame carrier drift in the role of the jitter term."""
    return int(math.ceil((WINDOW_BINS * df + DRIFT_FACTOR * float(k) * float(drift_rev_s)) / df))


def order_geometry(
    carriers: np.ndarray, drift: np.ndarray, df: float, rotor: int
) -> dict[str, Any]:
    """The order range of one rotor at one analysis length, with WHY it ends.

    ``k_sep_min``: below it the four rotors' lines are inside 2 bins of each
    other and no per-rotor line exists. ``k_drift_max``: above it the
    ``2 bins + 0.3 k drift`` window is wider than half the line spacing, so the
    line has swept into its neighbour inside one frame. ``k_band``: the
    30-7900 Hz observation band, with room for the +-1.5 rev/s search.
    """
    f_mean = float(carriers[rotor].mean())
    others = [float(carriers[r].mean()) for r in range(carriers.shape[0]) if r != rotor]
    gap = min(abs(f_mean - o) for o in others) if others else float("inf")
    drift_med = float(np.median(drift[rotor]))
    f_lo = float(carriers[rotor].min()) - REG_HALF_WIDTH
    f_hi = float(carriers[rotor].max()) + REG_HALF_WIDTH
    k_band_lo = max(1, int(math.ceil(BAND_LO / max(f_lo, 1e-6))))
    k_band_hi = int(math.floor(BAND_HI / max(f_hi, 1e-6)))
    k_sep_min = max(1, int(math.ceil(WINDOW_BINS * df / gap))) if math.isfinite(gap) else 1
    budget = WINDOW_SPACING_FRACTION * f_mean - WINDOW_BINS * df
    if drift_med <= 0.0:
        k_drift_max = k_band_hi
    elif budget <= 0.0:
        k_drift_max = 0
    else:
        k_drift_max = int(math.floor(budget / (DRIFT_FACTOR * drift_med)))
    k_lo, k_hi = max(k_band_lo, k_sep_min), min(k_band_hi, k_drift_max)
    ks = np.arange(k_lo, k_hi + 1, dtype=np.int64) if k_hi >= k_lo else np.zeros(0, dtype=np.int64)
    return dict(
        carrier_mean_rev_s=_r(f_mean, 4),
        carrier_min_rev_s=_r(float(carriers[rotor].min()), 4),
        carrier_max_rev_s=_r(float(carriers[rotor].max()), 4),
        drift_median_rev_s=_r(drift_med, 4),
        drift_max_rev_s=_r(float(drift[rotor].max()), 4),
        nearest_rotor_gap_rev_s=_r(gap, 4),
        bin_hz=_r(df, 5),
        k_sep_min=int(k_sep_min),
        k_drift_max=int(k_drift_max),
        k_band=[int(k_band_lo), int(k_band_hi)],
        k_used=[int(k_lo), int(k_hi)] if ks.size else None,
        n_orders_used=int(ks.size),
        n_orders_unresolvable_separability=int(max(0, k_sep_min - k_band_lo)),
        n_orders_unresolvable_drift=int(max(0, k_band_hi - k_drift_max)),
        _ks=ks,
    )


# ── the registration score and the per-order line table ─────────────────────


def _peak_floor(
    pm: np.ndarray, med: np.ndarray, df: float, k: int, half: int, centres: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(peak, floor)`` power of order ``k`` at ``centres`` (n_frames, n_delta)."""
    size = 2 * min(int(half), pm.shape[-1] // 4) + 1
    mx = maximum_filter1d(pm, size=size, axis=-1, mode="nearest")
    idx = np.clip(np.rint(centres / df).astype(np.int64), 0, pm.shape[-1] - 1)
    rows = np.arange(pm.shape[0])[:, None]
    return mx[rows, idx], med[rows, idx]


def score_curve(
    pm: np.ndarray,
    med: np.ndarray,
    df: float,
    carriers: np.ndarray,
    drift_med: float,
    ks: np.ndarray,
    deltas: np.ndarray,
    *,
    k_offset: float = 0.0,
) -> dict[str, np.ndarray]:
    """``S(delta)`` summed over orders AND frames, in two forms.

    ``raw``: the assignment's literal ``sum_k 10 log10 max P``. ``snr``: the
    same maximum over the local-median floor, capped at
    :data:`SNR_CAP_DB` per order, which is the robust form
    ``bench_diag.score_curve`` settled on and the one the percentage gain of
    the verdict is computed from (it is non-negative, so a ratio means
    something). ``k_offset = 0.5`` sweeps a HALF-ORDER comb, which carries the
    same number of orders, the same window widths and the same autocorrelation
    in ``delta`` but no lines: the null the peak height is judged against.
    """
    raw = np.zeros(deltas.size)
    snr = np.zeros(deltas.size)
    for k in ks:
        half = half_width_bins(int(k), df, drift_med)
        centres = (float(k) + float(k_offset)) * (carriers[:, None] + deltas[None, :])
        peak, floor = _peak_floor(pm, med, df, int(k), half, centres)
        raw += db(peak).sum(axis=0)
        snr += np.minimum(db(peak) - db(floor), SNR_CAP_DB).sum(axis=0)
    n = float(pm.shape[0])
    return dict(raw=raw / n, snr=snr / n)


def sign_test(label_snr: list[Any], null_snr: list[Any]) -> dict[str, Any]:
    """Is the per-order line SNR at the label ABOVE the half-order null?

    A paired sign test over orders. Under "no comb at the label" the label wins
    half the orders and the mean excess is 0; ``z_sign >= 3`` is the detection
    this check calls a comb. Nothing here depends on the absolute value of the
    peak-over-median estimator, which is biased by the window width and by the
    eight-microphone average and so cannot be compared to 6 dB on its own.
    """
    a = np.asarray([float(v) for v in label_snr if v is not None])
    b = np.asarray([float(v) for v in null_snr if v is not None])
    n = int(min(a.size, b.size))
    if n == 0:
        return dict(n_orders=0, z_sign=None, mean_excess_db=None, frac_above=None)
    d = a[:n] - b[:n]
    wins = int((d > 0.0).sum())
    return dict(
        n_orders=n,
        n_label_above_null=wins,
        frac_above=_r(wins / n, 3),
        z_sign=_r((wins - 0.5 * n) / math.sqrt(0.25 * n), 2),
        mean_excess_db=_r(float(d.mean()), 3),
        median_excess_db=_r(float(np.median(d)), 3),
    )


def line_table(
    pm: np.ndarray,
    med: np.ndarray,
    df: float,
    carriers: np.ndarray,
    drift_med: float,
    ks: np.ndarray,
    delta: float,
    *,
    k_offset: float = 0.0,
) -> dict[str, Any]:
    """Per-order line SNR at carrier ``f_r + delta``: MEDIAN over frames.

    ``k_offset = 0.5`` reads the same statistic HALF an order off every line,
    where the model puts no line at all: the empirical null this estimator
    carries. Mic-averaging over eight channels makes a peak-over-local-median
    of ~3 dB the noise expectation, so a raw 6 dB count means nothing without
    it.
    """
    snr: list[float] = []
    excess = 0.0
    for k in ks:
        half = half_width_bins(int(k), df, drift_med)
        centres = (float(k) + float(k_offset)) * (carriers[:, None] + float(delta))
        peak, floor = _peak_floor(pm, med, df, int(k), half, centres)
        snr.append(float(np.median(db(peak) - db(floor))))
        excess += float(np.mean(np.maximum(peak[:, 0] - floor[:, 0], 0.0)))
    arr = np.asarray(snr) if snr else np.zeros(0)
    return dict(
        delta_rev_s=_r(delta, 5),
        k_offset=float(k_offset),
        k=[int(v) for v in ks],
        snr_db=[_r(v, 2) for v in arr],
        n_orders=int(arr.size),
        n_above_6db=int((arr >= LINE_SNR_THRESHOLD_DB).sum()),
        snr_max_db=_r(float(arr.max()), 2) if arr.size else None,
        snr_median_db=_r(float(np.median(arr)), 2) if arr.size else None,
        line_excess_power=float(excess),
    )


def peak_stats(s: np.ndarray) -> dict[str, Any]:
    """Is ``S(delta)`` a PEAK at ``delta*`` or a flat noise curve?

    ``z``: how far the argmax stands above the curve's own median in robust
    sd. A genuine registration offset makes one narrow peak (large ``z``); a
    label that already sits on the lines, or a window with no lines at all,
    gives a flat curve whose argmax is wherever the noise happens to be
    highest, and whose ``delta*`` then lands on the grid edge as often as not.
    """
    j = int(np.argmax(s))
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med)))
    sd = 1.4826 * mad
    return dict(
        S_median=_r(med, 2),
        S_robust_sd=_r(sd, 3),
        S_peak_z=_r((float(s[j]) - med) / sd, 2) if sd > 0.0 else None,
        delta_star_at_grid_edge=bool(j in (0, s.size - 1)),
    )


def register_rotor(
    pm: np.ndarray,
    med: np.ndarray,
    df: float,
    carriers: np.ndarray,
    geom: dict[str, Any],
    deltas: np.ndarray,
) -> dict[str, Any]:
    """(b) + (c) for one rotor at one analysis length."""
    ks = geom["_ks"]
    if ks.size == 0:
        return dict(unresolvable=True, reason="no order carries a separable, undrifted line")
    drift_med = float(geom["drift_median_rev_s"] or 0.0)
    curves = score_curve(pm, med, df, carriers, drift_med, ks, deltas)
    i0 = int(np.argmin(np.abs(deltas)))
    out: dict[str, Any] = dict(unresolvable=False, n_orders=int(ks.size))
    for tag, s in curves.items():
        j = int(np.argmax(s))
        gain = float(s[j] - s[i0])
        out[tag] = dict(
            delta_star_rev_s=_r(float(deltas[j]), 4),
            S_at_label=_r(float(s[i0]), 2),
            S_at_delta_star=_r(float(s[j]), 2),
            S_gain=_r(gain, 2),
            S_gain_pct=(_r(100.0 * gain / abs(float(s[i0])), 2) if s[i0] != 0.0 else None),
            S_gain_per_order_db=_r(gain / float(ks.size), 3),
        ) | peak_stats(s)
    delta_star = float(deltas[int(np.argmax(curves["snr"]))])
    null = score_curve(pm, med, df, carriers, drift_med, ks, deltas, k_offset=0.5)
    out["snr_null"] = peak_stats(null["snr"]) | dict(
        S_at_label=_r(float(null["snr"][i0]), 2),
        S_gain=_r(float(null["snr"].max() - null["snr"][i0]), 2),
        S_gain_pct=(
            _r(
                100.0 * float(null["snr"].max() - null["snr"][i0]) / abs(float(null["snr"][i0])),
                2,
            )
            if null["snr"][i0] != 0.0
            else None
        ),
    )
    out["lines_at_label"] = line_table(pm, med, df, carriers, drift_med, ks, 0.0)
    out["lines_at_delta_star"] = line_table(pm, med, df, carriers, drift_med, ks, delta_star)
    out["lines_at_half_order_null"] = line_table(
        pm, med, df, carriers, drift_med, ks, 0.0, k_offset=0.5
    )
    out["label_vs_null"] = sign_test(
        out["lines_at_label"]["snr_db"], out["lines_at_half_order_null"]["snr_db"]
    )
    out["delta_star_vs_null"] = sign_test(
        out["lines_at_delta_star"]["snr_db"], out["lines_at_half_order_null"]["snr_db"]
    )
    e0 = out["lines_at_label"]["line_excess_power"]
    e1 = out["lines_at_delta_star"]["line_excess_power"]
    out["line_excess_gain_db"] = _r(float(db(e1) - db(e0)), 2) if e0 > 0.0 else None
    out["delta_star_bins_at_k2"] = _r(2.0 * abs(delta_star) / df, 3)
    out["_curve"] = curves["snr"]
    out["_curve_null"] = null["snr"]
    return out


# ── the order-tracked cross-check (drift-immune) ────────────────────────────


def order_track(audio: np.ndarray, label: np.ndarray, rotor: int) -> dict[str, Any]:
    """Resample the window onto rotor ``rotor``'s own uniform-revolution grid.

    Returns the mic-mean spectrum on an ORDER axis (cycles per revolution) plus
    its bin width. Every order of that rotor is then a constant-frequency line
    over the whole 4 s regardless of how far the carrier drifted, and a label
    bias ``delta`` shows up at order ``k`` as an offset ``k delta / f_mean``.

    The audio is upsampled before the non-uniform interpolation because linear
    interpolation of a 16 kHz signal that carries content at 8 kHz would cost
    several dB there. The resampled series has ``n_out`` samples spaced
    ``d_theta`` revolutions, so its ``rfft`` bins sit at ``j / (n_out
    d_theta) = j / Phi`` cycles per revolution with ``Phi`` the revolutions the
    window covers: the ORDER bin is ``1 / Phi``, which is ``1 / (f T)`` — the
    same 1/T = 0.25 Hz resolution the 4 s window has, now available at every
    order at once.
    """
    t_len = int(audio.shape[1])
    up = resample_poly(audio, ORDER_TRACK_UP, 1, axis=-1)
    sr_up = float(SU.SR * ORDER_TRACK_UP)
    t_up = np.arange(up.shape[1]) / sr_up
    rev = np.cumsum(label[rotor]) / float(SU.SR)
    rev_up = np.interp(t_up, np.arange(t_len) / float(SU.SR), rev)
    n_out = 2 * t_len
    theta = np.linspace(float(rev_up[0]), float(rev_up[-1]), n_out, endpoint=False)
    d_theta = float(theta[1] - theta[0])
    t_at = np.interp(theta, rev_up, t_up)
    y = np.stack([np.interp(t_at, t_up, row) for row in up])
    w = np.hanning(n_out + 1)[:n_out]
    power = (np.abs(np.fft.rfft(y * w, axis=-1)) ** 2 / float((w**2).sum())).mean(axis=0)
    revolutions = float(n_out) * d_theta
    return dict(
        power=power[None, :],
        d_order=1.0 / revolutions,
        order_nyquist=0.5 / d_theta,
        n_revolutions=revolutions,
        f_mean=float(label[rotor].mean()),
    )


def register_order_tracked(
    tracked: dict[str, Any], geom: dict[str, Any], deltas: np.ndarray
) -> dict[str, Any]:
    """(b) + (c) on the order-tracked spectrum, with ``delta`` still in rev/s."""
    ks = geom["_ks"]
    if ks.size == 0:
        return dict(unresolvable=True, reason="no separable order")
    pm = np.asarray(tracked["power"])
    d_order = float(tracked["d_order"])
    f_mean = float(tracked["f_mean"])
    med = local_floor(pm, d_order * f_mean)
    # A delta of rev/s maps to k*delta/f_mean in orders; the search window is
    # 2 order bins plus the intrinsic line width allowance.
    half = int(math.ceil((WINDOW_BINS * d_order + ORDER_TRACK_WIDTH_HZ / f_mean) / d_order))
    eps = deltas / f_mean
    raw = np.zeros(deltas.size)
    snr = np.zeros(deltas.size)
    snr_null = np.zeros(deltas.size)
    for k in ks:
        for offset, acc in ((0.0, None), (0.5, snr_null)):
            centres = (float(k) + offset) * (1.0 + eps)[None, :]
            peak, floor = _peak_floor(pm, med, d_order, int(k), half, centres)
            capped = np.minimum(db(peak) - db(floor), SNR_CAP_DB)[0]
            if acc is None:
                raw += db(peak)[0]
                snr += capped
            else:
                acc += capped
    i0 = int(np.argmin(np.abs(deltas)))
    j = int(np.argmax(snr))
    delta_star = float(deltas[j])

    def lines(delta: float, k_offset: float = 0.0) -> dict[str, Any]:
        vals: list[float] = []
        excess = 0.0
        for k in ks:
            centres = np.array([[(float(k) + k_offset) * (1.0 + delta / f_mean)]])
            peak, floor = _peak_floor(pm, med, d_order, int(k), half, centres)
            vals.append(float(db(peak)[0, 0] - db(floor)[0, 0]))
            excess += float(max(peak[0, 0] - floor[0, 0], 0.0))
        arr = np.asarray(vals)
        return dict(
            delta_rev_s=_r(delta, 5),
            k_offset=float(k_offset),
            k=[int(v) for v in ks],
            snr_db=[_r(v, 2) for v in arr],
            n_orders=int(arr.size),
            n_above_6db=int((arr >= LINE_SNR_THRESHOLD_DB).sum()),
            snr_max_db=_r(float(arr.max()), 2),
            snr_median_db=_r(float(np.median(arr)), 2),
            line_excess_power=float(excess),
        )

    at_label, at_star = lines(0.0), lines(delta_star)
    null_lines = lines(0.0, 0.5)
    e0 = at_label["line_excess_power"]
    # The SHARP resolution the 4 s of coherence actually carries: one order bin
    # at k = 2 is a carrier offset of d_order * f_mean / 2 rev/s. The search
    # window above is deliberately wider (it allows an intrinsic line width),
    # but the verdict's "delta* below a bin at k=2" test uses the bin.
    bin_rev_s_at_k2 = d_order * f_mean / 2.0
    # In-band power OF THE SAME (order-tracked) spectrum, so the discrete-line
    # excess and the total can be divided: the fraction of the window's band
    # power that sits in this rotor's resolved lines.
    orders_axis = np.arange(pm.shape[-1]) * d_order
    in_band = (orders_axis >= BAND_LO / f_mean) & (orders_axis <= BAND_HI / f_mean)
    band = float(pm[0, in_band].sum())
    return dict(
        unresolvable=False,
        n_orders=int(ks.size),
        n_revolutions=_r(tracked["n_revolutions"], 2),
        order_bin=_r(d_order, 6),
        window_half_orders=_r(half * d_order, 5),
        bin_rev_s_at_k2=_r(bin_rev_s_at_k2, 4),
        window_half_rev_s_at_k2=_r(half * d_order * f_mean / 2.0, 3),
        delta_star_bins_at_k2=_r(abs(delta_star) / bin_rev_s_at_k2, 3),
        raw=dict(
            delta_star_rev_s=_r(float(deltas[int(np.argmax(raw))]), 4),
            S_at_label=_r(float(raw[i0]), 2),
            S_at_delta_star=_r(float(raw.max()), 2),
            S_gain=_r(float(raw.max() - raw[i0]), 2),
        )
        | peak_stats(raw),
        snr=dict(
            delta_star_rev_s=_r(delta_star, 4),
            S_at_label=_r(float(snr[i0]), 2),
            S_at_delta_star=_r(float(snr[j]), 2),
            S_gain=_r(float(snr[j] - snr[i0]), 2),
            S_gain_pct=(
                _r(100.0 * float(snr[j] - snr[i0]) / abs(float(snr[i0])), 2)
                if snr[i0] != 0.0
                else None
            ),
            S_gain_per_order_db=_r(float(snr[j] - snr[i0]) / float(ks.size), 3),
        )
        | peak_stats(snr),
        snr_null=peak_stats(snr_null)
        | dict(
            S_at_label=_r(float(snr_null[i0]), 2),
            S_gain=_r(float(snr_null.max() - snr_null[i0]), 2),
            S_gain_pct=(
                _r(100.0 * float(snr_null.max() - snr_null[i0]) / abs(float(snr_null[i0])), 2)
                if snr_null[i0] != 0.0
                else None
            ),
        ),
        lines_at_label=at_label,
        lines_at_delta_star=at_star,
        lines_at_half_order_null=null_lines,
        label_vs_null=sign_test(at_label["snr_db"], null_lines["snr_db"]),
        delta_star_vs_null=sign_test(at_star["snr_db"], null_lines["snr_db"]),
        line_excess_gain_db=(
            _r(float(db(at_star["line_excess_power"]) - db(e0)), 2) if e0 > 0.0 else None
        ),
        line_excess_over_null_db=(
            _r(float(db(e0) - db(null_lines["line_excess_power"])), 2)
            if e0 > 0.0 and null_lines["line_excess_power"] > 0.0
            else None
        ),
        band_power=band,
        line_excess_over_band_db=_r(float(db(e0) - db(band)), 2) if e0 > 0.0 else None,
        null_excess_over_band_db=(
            _r(float(db(null_lines["line_excess_power"]) - db(band)), 2)
            if null_lines["line_excess_power"] > 0.0
            else None
        ),
        _curve=snr,
        _curve_null=snr_null,
    )


# ── the per-window run ──────────────────────────────────────────────────────


def band_power(pm: np.ndarray, df: float) -> float:
    freqs = np.arange(pm.shape[-1]) * df
    sel = (freqs >= BAND_LO) & (freqs <= BAND_HI)
    return float(pm[:, sel].sum(axis=-1).mean())


def run_window(
    support: GT.ScoredSupport,
    rps_key: str,
    *,
    audio: np.ndarray | None = None,
    label: np.ndarray | None = None,
) -> dict[str, Any]:
    if audio is None or label is None:
        audio, label = load_window(support, rps_key)
    deltas = np.arange(-REG_HALF_WIDTH, REG_HALF_WIDTH + 1e-9, REG_STEP)
    n_rotors = int(label.shape[0])
    out: dict[str, Any] = dict(
        recording=support.recording,
        key=support.key,
        start_s=float(support.start_s),
        duration_s=float(support.duration_s),
        rps_key=rps_key,
        sr=int(SU.SR),
        n_mics=int(audio.shape[0]),
        n_rotors=n_rotors,
        grid=dict(half_width_rev_s=REG_HALF_WIDTH, step_rev_s=REG_STEP, n=int(deltas.size)),
        label=dict(
            mean_rev_s=[_r(float(label[r].mean()), 4) for r in range(n_rotors)],
            drift_rev_s=[_r(float(label[r].max() - label[r].min()), 4) for r in range(n_rotors)],
            end_minus_start_rev_s=[
                _r(float(label[r, -1] - label[r, 0]), 4) for r in range(n_rotors)
            ],
            sorted_gaps_rev_s=[_r(v, 4) for v in np.diff(np.sort(label.mean(axis=1)))],
        ),
        analyses={},
    )
    frame_curves: dict[str, list[np.ndarray]] = {}
    frame_nulls: dict[str, list[np.ndarray]] = {}
    for an in ANALYSES:
        starts = an.starts(int(audio.shape[1]))
        n = min(an.n, int(audio.shape[1]))
        pm = frame_power(audio, starts, n)
        df = float(SU.SR) / float(n)
        med = local_floor(pm, df)
        carriers, drift = frame_labels(label, starts, n)
        rotors: list[dict[str, Any]] = []
        for r in range(n_rotors):
            geom = order_geometry(carriers, drift, df, r)
            row = register_rotor(pm, med, df, carriers[r], geom, deltas)
            frame_curves.setdefault(an.name, []).append(
                np.asarray(row.pop("_curve", np.zeros(deltas.size)))
            )
            frame_nulls.setdefault(an.name, []).append(
                np.asarray(row.pop("_curve_null", np.zeros(deltas.size)))
            )
            rotors.append({k: v for k, v in geom.items() if k != "_ks"} | row)
        out["analyses"][an.name] = dict(
            n=int(n),
            hop=(None if an.hop is None else int(an.hop)),
            bin_hz=_r(df, 5),
            n_frames=int(starts.size),
            band_power_db=_r(float(db(band_power(pm, df))), 2),
            rotors=rotors,
        )
    tracked_rows: list[dict[str, Any]] = []
    curves: list[np.ndarray] = []
    curves_null: list[np.ndarray] = []
    # Order geometry for the order-tracked read uses the WINDOW carrier and no
    # drift term (the resampling removed it).
    carriers_w = label.mean(axis=1)[:, None]
    zero_drift = np.zeros_like(carriers_w)
    for r in range(n_rotors):
        tracked = order_track(audio, label, r)
        d_order = float(tracked["d_order"])
        geom = order_geometry(carriers_w, zero_drift, d_order * float(tracked["f_mean"]), r)
        row = register_order_tracked(tracked, geom, deltas)
        curves.append(np.asarray(row.pop("_curve", np.zeros(deltas.size))))
        curves_null.append(np.asarray(row.pop("_curve_null", np.zeros(deltas.size))))
        tracked_rows.append({k: v for k, v in geom.items() if k != "_ks"} | row)
    out["order_tracked"] = dict(
        upsample=ORDER_TRACK_UP, line_width_allowance_hz=ORDER_TRACK_WIDTH_HZ, rotors=tracked_rows
    )
    out["_curves"] = dict(
        deltas=deltas,
        order_tracked=curves,
        order_tracked_null=curves_null,
        frames=frame_curves,
        frames_null=frame_nulls,
        label_t=np.arange(0, int(label.shape[1]), max(1, int(label.shape[1]) // 400))
        / float(SU.SR),
        label_track=label[:, :: max(1, int(label.shape[1]) // 400)],
    )
    return out


# ── verdict and the comb-level implication ──────────────────────────────────


#: A comb is called DETECTED AT THE LABEL when the per-order line SNR there
#: beats the half-order null by this many sign-test sd over the window's
#: orders. ``delta = 0`` is not chosen by the data, so this statistic carries
#: no selection bias and 3 sd is the usual bar.
DETECT_Z = 3.0
#: The bar for the SAME statistic read at ``delta*``, which IS chosen by the
#: data and so is biased upward. Calibrated on the self-test's own null: with
#: no comb present (the -36 dB injection) the selected z reaches 3.6, and on
#: the real windows 4.2, while the self-test's genuine 0.4 rev/s-offset comb
#: gives 8.3-9.5. 5 sd separates them with room on both sides.
DETECT_Z_SELECTED = 5.0


def verdict(row: dict[str, Any]) -> dict[str, Any]:
    """REGISTERED / MISREGISTERED / NO_COMB per window x rotor.

    The assignment's test is "``|delta*|`` below a bin at ``k = 2`` and ``S``
    gain < 10 %". Two things had to be settled before it can be applied.

    *Which bin.* At the likelihood's own 2048-point frame one bin at ``k = 2``
    is 3.9 rev/s, WIDER than the whole +-1.5 rev/s search, so every offset
    passes the first half vacuously. The ORDER-TRACKED read of the same 4 s
    carries the resolution the data really has (one order bin at ``k = 2`` is
    ~0.10 rev/s) without the drift smear that ruins a plain 4 s periodogram, so
    the verdict is taken there and the likelihood-frame numbers are reported
    beside it.

    *Is there anything to register.* A binary REGISTERED/MISREGISTERED assumes
    the window HOLDS a comb and only asks where it is. If the line SNR at the
    label is indistinguishable from the half-order null and ``S(delta)`` has no
    peak above its own null, then no offset finds a comb and the label is not
    shown to be wrong — it is shown that there is nothing there. That is
    ``NO_COMB``, and it is a different implication for the comb level than
    either of the other two, so it is reported as its own verdict rather than
    folded into MISREGISTERED by an argmax that is chasing noise.
    """
    tracked = row["order_tracked"]["rotors"]
    frames = row["analyses"][VERDICT_ANALYSIS]["rotors"]
    per_rotor: list[dict[str, Any]] = []
    for r, rot in enumerate(tracked):
        if rot.get("unresolvable"):
            per_rotor.append(dict(rotor=r, verdict="UNRESOLVABLE", reason=rot.get("reason")))
            continue
        s, null = rot["snr"], rot["snr_null"]
        cmp_, cmp_star = rot["label_vs_null"], rot["delta_star_vs_null"]
        bins = float(rot["delta_star_bins_at_k2"] or 0.0)
        pct = s["S_gain_pct"]
        z_sign, z_star = cmp_["z_sign"], cmp_star["z_sign"]
        z_peak, z_peak_null = s["S_peak_z"], null["S_peak_z"]
        detected = z_sign is not None and float(z_sign) >= DETECT_Z
        # A comb AWAY from the label has to clear the same KIND of bar the
        # label does — its per-order lines must beat the half-order null by a
        # sign-test margin — but at the higher, selection-corrected
        # :data:`DETECT_Z_SELECTED`, because delta* is chosen by the data. A
        # fixed threshold on the S gain would not do at all: the half-order
        # null's own gain reaches 16 % on these windows, so 10 % is inside the
        # noise. The self-test's planted +0.4 rev/s comb clears this bar at
        # z = 8.3-9.5 with delta* recovered to 0.054 rev/s.
        offset_comb = z_star is not None and float(z_star) >= DETECT_Z_SELECTED
        ok_delta = bins < 1.0
        ok_gain = pct is not None and float(pct) < 10.0
        if detected and ok_delta and ok_gain:
            call = "REGISTERED"
        elif detected or (offset_comb and not ok_delta):
            call = "MISREGISTERED"
        else:
            call = "NO_COMB"
        frame = frames[r]
        per_rotor.append(
            dict(
                rotor=r,
                n_orders=rot["n_orders"],
                delta_star_rev_s=s["delta_star_rev_s"],
                bin_rev_s_at_k2=rot["bin_rev_s_at_k2"],
                delta_star_bins_at_k2=_r(bins, 3),
                delta_star_at_grid_edge=s["delta_star_at_grid_edge"],
                S_gain_db=s["S_gain"],
                S_gain_pct=pct,
                S_peak_z=z_peak,
                S_peak_z_null=z_peak_null,
                z_sign_delta_star_vs_null=z_star,
                S_gain_pct_null=null["S_gain_pct"],
                z_sign_label_vs_null=z_sign,
                mean_excess_db_label_vs_null=cmp_["mean_excess_db"],
                n_above_6db_label=rot["lines_at_label"]["n_above_6db"],
                n_above_6db_delta_star=rot["lines_at_delta_star"]["n_above_6db"],
                n_above_6db_null=rot["lines_at_half_order_null"]["n_above_6db"],
                snr_median_db_label=rot["lines_at_label"]["snr_median_db"],
                snr_median_db_null=rot["lines_at_half_order_null"]["snr_median_db"],
                line_excess_gain_db=rot["line_excess_gain_db"],
                line_excess_over_band_db=rot["line_excess_over_band_db"],
                null_excess_over_band_db=rot["null_excess_over_band_db"],
                frame_S_gain_pct=(
                    None if frame.get("unresolvable") else frame["snr"]["S_gain_pct"]
                ),
                frame_z_sign=(
                    None if frame.get("unresolvable") else frame["label_vs_null"]["z_sign"]
                ),
                frame_n_above_6db_label=(
                    None if frame.get("unresolvable") else frame["lines_at_label"]["n_above_6db"]
                ),
                frame_n_above_6db_null=(
                    None
                    if frame.get("unresolvable")
                    else frame["lines_at_half_order_null"]["n_above_6db"]
                ),
                comb_detected=bool(detected),
                verdict=call,
            )
        )
    named = [v for v in per_rotor if v["verdict"] != "UNRESOLVABLE"]
    calls = {v["verdict"] for v in named}
    return dict(
        analysis="order_tracked_4s",
        cross_check_analysis=VERDICT_ANALYSIS,
        detect_z=DETECT_Z,
        detect_z_selected=DETECT_Z_SELECTED,
        n_rotors_comb_detected=sum(1 for v in named if v["comb_detected"]),
        per_rotor=per_rotor,
        verdict=(
            "UNRESOLVABLE"
            if not named
            else "MISREGISTERED"
            if "MISREGISTERED" in calls
            else "REGISTERED"
            if "REGISTERED" in calls
            else "NO_COMB"
        ),
    )


def _median_excess(cases: list[dict[str, Any]]) -> float | None:
    vals = [
        float(v["mean_excess_db"] if "mean_excess_db" in v else v["mean_excess_db_label_vs_null"])
        for v in cases
        if (v.get("mean_excess_db") or v.get("mean_excess_db_label_vs_null")) is not None
    ]
    return float(np.median(vals)) if vals else None


def comb_level_bound(rows: list[dict[str, Any]], st: dict[str, Any] | None) -> dict[str, Any]:
    """The comb band level the five windows' spectra actually support, in dB.

    The self-test's level sweep is a calibration curve: injected comb band
    level (dB re the window's own band power) against the mean per-order line
    SNR excess over the half-order null, which is the statistic the detection
    rests on and the only one that goes to ZERO when there is no comb (the
    absolute line excess does not — the max-over-window estimator keeps a
    positive bias that its own null only estimates to within its noise).
    Reading the real windows' median measurement off that curve gives the
    level, or an upper bound at the quietest calibrated point when the
    measurement falls below it.
    """
    observed = [
        v for row in rows for v in row["verdict"]["per_rotor"] if v["verdict"] != "UNRESOLVABLE"
    ]
    obs = _median_excess(observed)
    out: dict[str, Any] = dict(
        observed_median_excess_db=_r(obs, 3),
        observed_excess_db=[v["mean_excess_db_label_vs_null"] for v in observed],
        observed_median_z_sign=_r(
            float(np.median([float(v["z_sign_label_vs_null"]) for v in observed])), 2
        )
        if observed
        else None,
    )
    if st is None:
        return out
    curve = [
        (
            float(c["comb_level_db"]),
            _median_excess(c["per_rotor"]),
            float(np.median([float(v["z_sign"]) for v in c["per_rotor"]])),
            int(c["n_rotors_detected"]),
            int(c["n_rotors"]),
        )
        for c in st["level_sweep"]
        if c["per_rotor"]
    ]
    curve = [t for t in curve if t[1] is not None]
    out["calibration"] = [
        dict(
            level_db=lv,
            median_excess_db=_r(ex, 3),
            median_z_sign=_r(z, 2),
            n_detected=nd,
            n_rotors=nr,
        )
        for lv, ex, z, nd, nr in curve
    ]
    asc = sorted(((float(t[1] or 0.0), t[0]) for t in curve), key=lambda p: p[0])
    if obs is not None and asc:
        if obs < asc[0][0]:
            out["comb_level_upper_bound_db"] = _r(asc[0][1], 2)
            out["comb_level_db"] = None
        else:
            out["comb_level_db"] = _r(
                float(np.interp(obs, [p[0] for p in asc], [p[1] for p in asc])), 2
            )
            out["comb_level_upper_bound_db"] = None
    out["note"] = (
        "comb_level_db is the injected comb band level (dB re the window's own 30-7900 Hz "
        "power) whose mean per-order SNR excess over the half-order null matches the real "
        "windows' median; comb_level_upper_bound_db is set instead when the real measurement "
        "falls below the quietest calibrated injection"
    )
    return out


def comb_level_implication(
    rows: list[dict[str, Any]], st: dict[str, Any] | None = None
) -> dict[str, Any]:
    """What the registration numbers imply for the ~25 dB comb-level disagreement.

    The exact statement — re-evaluating the R2 floor-gain fit's Whittle
    objective at ``delta*`` with the comb gain profiled — needs one forward
    pass of ``experiments.noise_model.spectrum`` per grid point on a flight
    grid, and that module is being rewritten for R3 while this runs, so it is
    NOT cheap here. What IS available from the measurements above is the
    narrowband profiled-gain proxy: a global comb gain is pulled to whatever
    explains the measured discrete-line excess over the floor, so the recovery
    a re-registration could buy is ``10 log10 E(delta*) / E(0)`` with
    ``E`` the summed line excess power. The order-tracked read gives the
    best-case (fully drift-compensated, 4 s coherent) version of the same.
    """
    fit: dict[str, Any] = {}
    if R2_FIT.exists():
        payload = json.loads(R2_FIT.read_text())
        fit = dict(
            path=str(R2_FIT),
            schema=payload.get("schema"),
            comb_gain_db=_r(payload["params"]["profile"].get("comb_gain_db"), 3),
            init_comb_gain_db=_r((payload.get("diagnostics") or {}).get("init_comb_gain_db"), 3),
            floor_level_db=_r((payload.get("diagnostics") or {}).get("floor_level_db"), 3),
        )
    per_window: list[dict[str, Any]] = []
    for row in rows:
        an = row["analyses"][VERDICT_ANALYSIS]
        gains = [
            r["line_excess_gain_db"]
            for r in an["rotors"]
            if not r.get("unresolvable") and r.get("line_excess_gain_db") is not None
        ]
        tracked = [
            r["line_excess_gain_db"]
            for r in row["order_tracked"]["rotors"]
            if not r.get("unresolvable") and r.get("line_excess_gain_db") is not None
        ]
        keep = [r for r in row["order_tracked"]["rotors"] if not r.get("unresolvable")]
        excess = sum(float(r["lines_at_label"]["line_excess_power"]) for r in keep)
        band = float(np.mean([float(r["band_power"]) for r in keep])) if keep else 0.0
        keep_rows = [
            dict(
                line_excess_over_band_db=r["line_excess_over_band_db"],
                null_excess_over_band_db=r["null_excess_over_band_db"],
            )
            for r in keep
        ]
        per_window.append(
            dict(
                recording=row["recording"],
                rps_key=row["rps_key"],
                frame_line_excess_gain_db=[_r(v, 2) for v in gains],
                order_tracked_line_excess_gain_db=[_r(v, 2) for v in tracked],
                line_excess_over_band_db=[r["line_excess_over_band_db"] for r in keep],
                null_excess_over_band_db=[r["null_excess_over_band_db"] for r in keep],
                net_excess_over_band_db=net_excess(keep_rows),
                all_rotor_line_excess_over_band_db=(
                    _r(float(db(excess) - db(band)), 2) if excess > 0.0 and band > 0.0 else None
                ),
            )
        )
    flat = [
        float(v)
        for w in per_window
        for v in w["order_tracked_line_excess_gain_db"]
        if v is not None
    ]
    frame_flat = [
        float(v) for w in per_window for v in w["frame_line_excess_gain_db"] if v is not None
    ]
    return dict(
        r2_fit=fit,
        level_matched_shift_db=LEVEL_MATCHED_SHIFT_DB,
        disagreement_db=(
            _r(LEVEL_MATCHED_SHIFT_DB - float(fit["comb_gain_db"]), 2)
            if fit.get("comb_gain_db") is not None
            else None
        ),
        exact_whittle_reevaluation="not cheap: needs a spectrum.py flight forward pass per grid "
        "point, and spectrum/model are being rewritten for R3 concurrently",
        proxy="10 log10 sum_k line-excess(delta*) / sum_k line-excess(0)",
        line_content_note="line_excess_over_band_db is the fraction of the window's 30-7900 Hz "
        "power that sits in that rotor's resolved lines, measured on the order-tracked "
        "spectrum; null_excess_over_band_db is the same read half an order off every line, "
        "i.e. the estimator's own bias. The comb level the data supports is bounded by the "
        "difference of the two.",
        per_window=per_window,
        comb_level=comb_level_bound(rows, st),
        self_test_sensitivity_db=(None if st is None else st.get("sensitivity_db")),
        proxy_recovery_db=dict(
            frame_median=_r(float(np.median(frame_flat)), 2) if frame_flat else None,
            frame_max=_r(float(np.max(frame_flat)), 2) if frame_flat else None,
            order_tracked_median=_r(float(np.median(flat)), 2) if flat else None,
            order_tracked_max=_r(float(np.max(flat)), 2) if flat else None,
        ),
    )


# ── the self-test: can this estimator find a comb it KNOWS is there? ────────

#: Band-power levels of the injected comb, relative to the real window's own
#: 30-7900 Hz power. 0 dB is what the gate's band-level-matched arm renders
#: (comb band power = the real clip's band power); the sweep goes down until the
#: check loses the comb, which is what turns a null result into a BOUND on the
#: comb level the real windows can hold.
SELFTEST_LEVELS_DB = (0.0, -6.0, -12.0, -18.0, -24.0, -30.0, -36.0)
#: The carrier offset the self-test plants to prove ``delta*`` is recovered.
SELFTEST_OFFSET_REV_S = 0.4
SELFTEST_SEED = 0


def synthetic_comb(
    audio: np.ndarray, label: np.ndarray, *, delta_rev_s: float, level_db: float, seed: int
) -> np.ndarray:
    """``audio`` plus a comb on ``label + delta``, at ``level_db`` of band power.

    Every rotor gets orders 1..K with equal line power and a fixed random phase
    per line, the phase integrated from the label's own (drifting) carrier — so
    this is exactly the generative comb of the model, planted at a KNOWN
    offset. The same comb goes to every microphone, which is the easiest case
    for the mic-mean periodogram: if the estimator cannot find THIS, its null
    result on the real windows would mean nothing.
    """
    rng = np.random.default_rng(seed)
    t_len = int(audio.shape[1])
    comb = np.zeros(t_len)
    for r in range(int(label.shape[0])):
        f = label[r] + float(delta_rev_s)
        rev = 2.0 * np.pi * np.cumsum(f) / float(SU.SR)
        k_max = int(math.floor(BAND_HI / float(f.max())))
        for k in range(max(1, int(math.ceil(BAND_LO / float(f.min())))), k_max + 1):
            comb += np.cos(float(k) * rev + rng.uniform(0.0, 2.0 * np.pi))
    w = np.hanning(t_len + 1)[:t_len]
    spec = lambda x: np.abs(np.fft.rfft(x * w)) ** 2 / float((w**2).sum())  # noqa: E731
    freqs = np.fft.rfftfreq(t_len, d=1.0 / float(SU.SR))
    sel = (freqs >= BAND_LO) & (freqs <= BAND_HI)
    p_real = float(np.mean([spec(x)[sel].sum() for x in audio]))
    p_comb = float(spec(comb)[sel].sum())
    gain = math.sqrt(10.0 ** (level_db / 10.0) * p_real / max(p_comb, 1e-300))
    return audio + gain * comb[None, :]


def net_excess(per_rotor: list[dict[str, Any]]) -> float | None:
    """Median over rotors of the line excess MINUS the estimator's own null.

    ``line_excess_over_band_db`` is biased upward by the max-over-window
    estimator, and the half-order read measures exactly that bias on the same
    spectrum, so the difference IN POWER is the comb power the window actually
    holds, as a fraction of its 30-7900 Hz band power. This is the statistic
    the self-test calibrates and the real windows are read on.
    """
    vals: list[float] = []
    for v in per_rotor:
        a, b = v.get("line_excess_over_band_db"), v.get("null_excess_over_band_db")
        if a is None or b is None:
            continue
        net = 10.0 ** (float(a) / 10.0) - 10.0 ** (float(b) / 10.0)
        vals.append(float(db(net)) if net > 0.0 else -300.0)
    return _r(float(np.median(vals)), 2) if vals else None


def _selftest_case(
    support: GT.ScoredSupport,
    rps_key: str,
    audio: np.ndarray,
    label: np.ndarray,
    *,
    level_db: float,
    true_delta: float,
) -> dict[str, Any]:
    mixed = synthetic_comb(
        audio, label, delta_rev_s=true_delta, level_db=level_db, seed=SELFTEST_SEED
    )
    row = run_window(support, rps_key, audio=mixed, label=label)
    row["verdict"] = verdict(row)
    per_rotor = [
        dict(
            rotor=v["rotor"],
            verdict=v["verdict"],
            delta_star_rev_s=v["delta_star_rev_s"],
            delta_error_rev_s=_r(float(v["delta_star_rev_s"]) - true_delta, 4),
            z_sign=v["z_sign_label_vs_null"],
            z_sign_delta_star=v["z_sign_delta_star_vs_null"],
            mean_excess_db=v["mean_excess_db_label_vs_null"],
            S_peak_z=v["S_peak_z"],
            S_peak_z_null=v["S_peak_z_null"],
            S_gain_pct=v["S_gain_pct"],
            n_above_6db_label=v["n_above_6db_label"],
            n_above_6db_null=v["n_above_6db_null"],
            frame_z_sign=v["frame_z_sign"],
            frame_n_above_6db_label=v["frame_n_above_6db_label"],
            comb_detected=v["comb_detected"],
            line_excess_over_band_db=v["line_excess_over_band_db"],
            null_excess_over_band_db=v["null_excess_over_band_db"],
        )
        for v in row["verdict"]["per_rotor"]
        if v["verdict"] != "UNRESOLVABLE"
    ]
    return dict(
        comb_level_db=float(level_db),
        true_delta_rev_s=float(true_delta),
        verdict=row["verdict"]["verdict"],
        n_rotors=len(per_rotor),
        n_rotors_detected=sum(1 for v in per_rotor if v["comb_detected"]),
        max_abs_delta_error_rev_s=(
            _r(max(abs(float(v["delta_error_rev_s"])) for v in per_rotor), 4) if per_rotor else None
        ),
        net_excess_over_band_db=net_excess(per_rotor),
        per_rotor=per_rotor,
    )


def self_test(support: GT.ScoredSupport, rps_key: str) -> dict[str, Any]:
    """Calibrate the check: plant a comb of known level and offset, recover it.

    A null result ("no comb at the label") is only worth the sensitivity behind
    it, so the sweep runs the WHOLE check on the real window plus a synthetic
    comb at descending band levels. ``sensitivity_db`` is the quietest level at
    which every rotor is still detected; the real windows' measured
    ``z_sign`` sitting inside the null then bounds their comb below it. The
    offset case proves ``delta*`` is recovered when there IS something to
    register, so a scattered ``delta*`` on real data means "nothing there", not
    "estimator cannot localise".
    """
    audio, label = load_window(support, rps_key)
    levels = [
        _selftest_case(support, rps_key, audio, label, level_db=lvl, true_delta=0.0)
        for lvl in SELFTEST_LEVELS_DB
    ]
    detected = [c for c in levels if c["n_rotors_detected"] == c["n_rotors"] and c["n_rotors"]]
    sensitivity = min((c["comb_level_db"] for c in detected), default=None)
    offset = _selftest_case(
        support,
        rps_key,
        audio,
        label,
        level_db=SELFTEST_LEVELS_DB[0],
        true_delta=SELFTEST_OFFSET_REV_S,
    )
    # Three things must hold for a NO_COMB on real data to mean anything:
    # the loud planted comb is REGISTERED, the offset one is MISREGISTERED with
    # delta* recovered, and the injections too quiet to detect do NOT come back
    # MISREGISTERED — the last is what pins :data:`DETECT_Z_SELECTED`.
    no_false_positive = all(
        c["verdict"] != "MISREGISTERED" for c in levels if c["n_rotors_detected"] == 0
    )
    passed = bool(
        sensitivity is not None
        and levels[0]["verdict"] == "REGISTERED"
        and offset["verdict"] == "MISREGISTERED"
        and (offset["max_abs_delta_error_rev_s"] or 1e9) < 0.2
        and no_false_positive
    )
    return dict(
        recording=support.recording,
        rps_key=rps_key,
        levels_db=list(SELFTEST_LEVELS_DB),
        offset_rev_s=SELFTEST_OFFSET_REV_S,
        sensitivity_db=sensitivity,
        no_false_positive=bool(no_false_positive),
        level_sweep=levels,
        offset_case=offset,
        passed=passed,
        claim="a comb planted on the label at band level >= sensitivity_db is called REGISTERED "
        "with delta* = 0; the same comb planted 0.4 rev/s off the label is called "
        "MISREGISTERED with delta* recovered to better than 0.2 rev/s on every rotor; and an "
        "injection too quiet to detect is never called MISREGISTERED",
    )


# ── figures ─────────────────────────────────────────────────────────────────


def _save(fig: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return str(path)


def write_figures(rows: list[dict[str, Any]], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    primary = [r for r in rows if r["rps_key"] == LABEL_KEYS[0]]
    colours = ("#1b6ca8", "#c0392b", "#27ae60", "#8e44ad")
    written: list[str] = []

    fig, axes = plt.subplots(
        len(primary), 2, figsize=(12.5, 2.2 * len(primary)), sharex=True, squeeze=False
    )
    for i, row in enumerate(primary):
        deltas = row["_curves"]["deltas"]
        panels = (
            (
                0,
                row["analyses"][VERDICT_ANALYSIS]["rotors"],
                row["_curves"]["frames"][VERDICT_ANALYSIS],
                row["_curves"]["frames_null"][VERDICT_ANALYSIS],
            ),
            (
                1,
                row["order_tracked"]["rotors"],
                row["_curves"]["order_tracked"],
                row["_curves"]["order_tracked_null"],
            ),
        )
        for col, rots, curves, nulls in panels:
            ax = axes[i, col]
            for r, curve in enumerate(curves):
                if rots[r].get("unresolvable"):
                    continue
                ax.plot(deltas, curve, lw=0.9, color=colours[r % 4])
                ax.plot(deltas, nulls[r], lw=0.7, ls="--", color=colours[r % 4], alpha=0.45)
                ax.axvline(
                    float(rots[r]["snr"]["delta_star_rev_s"]),
                    color=colours[r % 4],
                    lw=0.7,
                    alpha=0.4,
                )
            ax.axvline(0.0, color="k", lw=1.0, ls=":")
            ax.grid(alpha=0.25)
        axes[i, 0].set_ylabel(
            row["recording"].replace("_nosource_room2", "") + "\nS (dB)", fontsize=8
        )
    axes[0, 0].set_title(f"S(delta), {VERDICT_ANALYSIS} (the likelihood's own frame)", fontsize=9)
    axes[0, 1].set_title("S(delta), order-tracked 4 s (drift-immune)", fontsize=9)
    for col in (0, 1):
        axes[-1, col].set_xlabel("label carrier offset delta (rev/s)")
    handles = [Line2D([], [], color=colours[r], lw=1.2, label=f"rotor {r}") for r in range(4)] + [
        Line2D([], [], color="0.3", lw=1.2, label="at the comb"),
        Line2D([], [], color="0.3", lw=1.0, ls="--", label="half-order null"),
        Line2D([], [], color="k", lw=1.0, ls=":", label="delta = 0 (the label)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "DREGON label registration: capped per-order SNR sum vs carrier offset\n"
        "(vertical coloured lines mark each rotor's argmax delta*)",
        y=1.0,
        fontsize=11,
    )
    written.append(_save(fig, out / "score_curves.png"))
    plt.close(fig)

    fig, axs = plt.subplots(
        len(primary), 1, figsize=(9.5, 1.9 * len(primary)), sharex=True, squeeze=False
    )
    axes2 = axs[:, 0]
    for i, row in enumerate(primary):
        ax = axes2[i]
        keep = [r for r in row["order_tracked"]["rotors"] if not r.get("unresolvable")]
        for tag, key, style in (
            ("at the label comb", "lines_at_label", dict(color="#1b6ca8", lw=1.1)),
            ("half-order null", "lines_at_half_order_null", dict(color="#c0392b", lw=1.1, ls="--")),
        ):
            acc: dict[int, list[float]] = {}
            for rot in keep:
                for k, v in zip(rot[key]["k"], rot[key]["snr_db"], strict=True):
                    if v is not None:
                        acc.setdefault(int(k), []).append(float(v))
            ks = sorted(acc)
            ax.plot(ks, [float(np.mean(acc[k])) for k in ks], label=tag, **style)
        ax.axhline(LINE_SNR_THRESHOLD_DB, color="k", lw=0.7, ls=":")
        zs = [float(r["label_vs_null"]["z_sign"]) for r in keep]
        ax.set_ylabel(row["recording"].replace("_nosource_room2", "") + "\nSNR (dB)", fontsize=8)
        ax.set_title(
            "z(label vs null) per rotor: " + ", ".join(f"{v:+.2f}" for v in zs), fontsize=7
        )
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7, ncol=2)
    axes2[-1].set_xlabel("order k")
    fig.suptitle(
        "Order-tracked per-order line SNR over the local floor, MEAN over the four rotors:\n"
        "at the label's comb against the same read half an order off every line",
        y=1.0,
        fontsize=11,
    )
    written.append(_save(fig, out / "line_snr.png"))
    plt.close(fig)

    fig, axs = plt.subplots(
        len(primary), 1, figsize=(9.0, 1.9 * len(primary)), sharex=True, squeeze=False
    )
    axes = axs[:, 0]
    for i, row in enumerate(primary):
        ax = axes[i]
        t = row["_curves"]["label_t"]
        track = row["_curves"]["label_track"]
        for r in range(row["n_rotors"]):
            ax.plot(t, track[r], lw=0.9, color=colours[r % 4], label=f"r{r}")
        drift = row["label"]["drift_rev_s"]
        ax.set_ylabel(row["recording"].replace("_nosource_room2", "") + "\nrev/s", fontsize=7)
        ax.set_title(
            "means "
            + ", ".join(f"{v:.2f}" for v in row["label"]["mean_rev_s"])
            + " rev/s; within-window drift "
            + ", ".join(f"{v:.1f}" for v in drift)
            + " rev/s",
            fontsize=7,
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6, ncol=4)
    axes[-1].set_xlabel("time in the 4 s score window (s)")
    fig.suptitle("Label carriers and their within-window drift", y=1.0)
    written.append(_save(fig, out / "labels.png"))
    plt.close(fig)
    return written


# ── findings ────────────────────────────────────────────────────────────────


def findings(payload: dict[str, Any], figures: list[str]) -> str:
    o: list[str] = []
    o.append("# R3 DREGON label registration: do the labels register the real comb lines?")
    o.append("")
    o.append(f"`{payload['schema']}`, git `{payload['git']}`. Read-only; laptop CPU.")
    o.append("")
    o.append(
        "Question (R3 plan, `#model-r3` round table): the DREGON frozen-comb fit puts the comb "
        f"at `comb_gain_db = {payload['comb_level']['r2_fit'].get('comb_gain_db')}` dB while the "
        f"band-level match the HPPNet gate needs is `+{LEVEL_MATCHED_SHIFT_DB}` dB — a "
        f"{payload['comb_level']['disagreement_db']} dB disagreement. If the label carrier does "
        "not sit on the real lines the likelihood never saw the comb and the quiet comb is an "
        "artefact; if it does, the quiet comb is honest and the gate disagrees with the spectrum."
    )
    o.append("")
    o.append("## What the label does inside a 4 s score window")
    o.append("")
    o.append("| window | mean carriers (rev/s) | within-window drift (rev/s) | sorted gaps |")
    o.append("|---|---|---|---|")
    for row in payload["windows"]:
        if row["rps_key"] != LABEL_KEYS[0]:
            continue
        o.append(
            f"| `{row['recording'].replace('_nosource_room2', '')}` | "
            + ", ".join(f"{v:.2f}" for v in row["label"]["mean_rev_s"])
            + " | "
            + ", ".join(f"{v:.2f}" for v in row["label"]["drift_rev_s"])
            + " | "
            + ", ".join(f"{v:.2f}" for v in row["label"]["sorted_gaps_rev_s"])
            + " |"
        )
    o.append("")
    o.append(
        "The two published tracks agree: the `rps_refined` and `motors_command` window means "
        "differ by at most "
        f"{payload['label_agreement']['max_abs_mean_diff_rev_s']} rev/s, so no verdict below "
        "depends on which one is called 'the label'."
    )
    o.append("")
    o.append("## Separability (d)")
    o.append("")
    o.append(
        "| analysis | bin (Hz) | frames | k separable from k | k unusable above (drift) | orders used |"
    )
    o.append("|---|---:|---:|---:|---:|---:|")
    row0 = next(r for r in payload["windows"] if r["rps_key"] == LABEL_KEYS[0])
    for name, an in row0["analyses"].items():
        rot = an["rotors"]
        o.append(
            f"| `{name}` | {an['bin_hz']} | {an['n_frames']} | "
            f"{max(r['k_sep_min'] for r in rot)} | "
            f"{min(r['k_drift_max'] for r in rot)} | "
            f"{min(r['n_orders_used'] for r in rot)}-{max(r['n_orders_used'] for r in rot)} |"
        )
    o.append("")
    o.append("## Registration per window x rotor")
    o.append("")
    o.append(
        "`n>6 dB` counts orders whose line clears the local median floor by 6 dB. The "
        "peak-over-median estimator is biased upward by the search window and by the "
        "eight-microphone average, so every count is given beside the SAME read taken half an "
        "order off every line (`@null`), where the model puts nothing. The decision statistic "
        "is the paired sign test between the two, `z`."
    )
    o.append("")
    for tag, getter in (
        (VERDICT_ANALYSIS, lambda r: r["analyses"][VERDICT_ANALYSIS]["rotors"]),
        ("order_tracked_4s", lambda r: r["order_tracked"]["rotors"]),
    ):
        o.append(f"### `{tag}`")
        o.append("")
        o.append(
            "| window | rotor | k used | delta* (rev/s) | S gain % | S gain % @null | "
            "n>6 dB @label | @delta* | @null | z @label | z @delta* | mean excess (dB) |"
        )
        o.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in payload["windows"]:
            if row["rps_key"] != LABEL_KEYS[0]:
                continue
            name = row["recording"].replace("_nosource_room2", "")
            for r, rot in enumerate(getter(row)):
                if rot.get("unresolvable"):
                    o.append(f"| `{name}` | {r} | 0 | unresolvable | | | | | | | | |")
                    continue
                s, cmp_ = rot["snr"], rot["label_vs_null"]
                o.append(
                    f"| `{name}` | {r} | {rot['n_orders']} | {s['delta_star_rev_s']} | "
                    f"{s['S_gain_pct']} | {rot['snr_null']['S_gain_pct']} | "
                    f"{rot['lines_at_label']['n_above_6db']} | "
                    f"{rot['lines_at_delta_star']['n_above_6db']} | "
                    f"{rot['lines_at_half_order_null']['n_above_6db']} | "
                    f"{cmp_['z_sign']} | {rot['delta_star_vs_null']['z_sign']} | "
                    f"{cmp_['mean_excess_db']} |"
                )
        o.append("")
    lows: list[str] = []
    for row in payload["windows"]:
        if row["rps_key"] != LABEL_KEYS[0]:
            continue
        pair: list[tuple[float, float]] = []
        for rot in row["order_tracked"]["rotors"]:
            if rot.get("unresolvable"):
                continue
            lab, nul = rot["lines_at_label"], rot["lines_at_half_order_null"]
            if 1 in lab["k"]:
                j = lab["k"].index(1)
                pair.append((float(lab["snr_db"][j]), float(nul["snr_db"][j])))
        if pair:
            lows.append(
                f"`{row['recording'].replace('_nosource_room2', '')}` "
                f"{np.mean([p[0] for p in pair]):.1f} vs {np.mean([p[1] for p in pair]):.1f}"
            )
    if lows:
        o.append(
            "One order does stand above the null: `k = 1`, the shaft rate itself, at a mean "
            "label-vs-null SNR of " + "; ".join(lows) + " dB. It decides nothing about "
            "registration and is included in the sums above only because the sign test weights "
            "every order equally: at `k = 1` the search window spans a carrier offset of "
            "+-4.5 rev/s, three times the whole grid, so that order cannot localise `delta` at "
            "all; and its line sits at 76-85 Hz where the floor's own steep low-frequency slope "
            "biases a +-30 Hz local median downward, while the null reads it at 115-128 Hz on "
            "the flat part. Whether it is a real shaft line or a slope artefact, it carries no "
            "comb: one order of ~95, and the frozen comb's power is in `k >= 2`."
        )
        o.append("")
    o.append("## Verdict")
    o.append("")
    o.append(
        "Taken on the order-tracked 4 s read, where one bin at `k = 2` is ~0.10 rev/s. At the "
        f"likelihood's `{VERDICT_ANALYSIS}` one bin at `k = 2` is 3.9 rev/s — wider than the "
        "whole search — so the `|delta*| < 1 bin` half of the test cannot discriminate there; "
        "its `S` gain and line counts are in the table above."
    )
    o.append("")
    o.append(
        "| window | verdict | max \\|delta*\\| (rev/s) | in k=2 bins | max S gain % | "
        "max z @label | max z @delta* | rotors with a comb |"
    )
    o.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in payload["windows"]:
        if row["rps_key"] != LABEL_KEYS[0]:
            continue
        v = row["verdict"]
        named = [x for x in v["per_rotor"] if x["verdict"] != "UNRESOLVABLE"]
        name = row["recording"].replace("_nosource_room2", "")
        if not named:
            o.append(f"| `{name}` | **{v['verdict']}** | | | | | | |")
            continue
        # Summarise over the rotors that DECIDED the window's verdict, so a
        # REGISTERED window is not described by the delta* of a rotor that
        # carries no comb at all.
        decided = [x for x in named if x["verdict"] == v["verdict"]] or named
        o.append(
            f"| `{name}` | **{v['verdict']}** ({len(decided)}/{len(named)} rotors) | "
            f"{max(abs(float(x['delta_star_rev_s'])) for x in decided):.3f} | "
            f"{max(float(x['delta_star_bins_at_k2']) for x in decided):.2f} | "
            f"{max(float(x['S_gain_pct']) for x in decided):.2f} | "
            f"{max(float(x['z_sign_label_vs_null']) for x in decided):.2f} | "
            f"{max(float(x['z_sign_delta_star_vs_null']) for x in decided):.2f} | "
            f"{v['n_rotors_comb_detected']}/{len(named)} |"
        )
    o.append("")
    o.append(
        f"Bars: a comb is DETECTED at the label at `z >= {DETECT_Z}` and at `delta*` at "
        f"`z >= {DETECT_Z_SELECTED}` (the second is selection-corrected — see the self-test). "
        "REGISTERED needs a detection at the label with `|delta*|` inside a `k = 2` bin and an "
        "`S` gain under 10 %; MISREGISTERED needs a detection off the label; NO_COMB is "
        "returned when no offset in +-1.5 rev/s finds a comb at all, which is a different "
        "implication for the comb level and so is not folded into MISREGISTERED by an argmax "
        "chasing noise."
    )
    o.append("")
    st = payload.get("self_test")
    if st:
        o.append("## Self-test: the sensitivity behind the null")
        o.append("")
        o.append(
            f"A null result is worth only the sensitivity behind it, so the whole check was re-run "
            f"on `{st['recording']}` plus a SYNTHETIC comb of the model's own form, planted on the "
            f"label at descending band levels (dB re that window's own 30-7900 Hz power; 0 dB is "
            f"what the gate's band-level-matched arm renders) and once at a deliberate "
            f"+{st['offset_rev_s']} rev/s offset. `passed = {st['passed']}`."
        )
        o.append("")
        o.append(
            "| injected comb level (dB) | planted delta (rev/s) | verdict | rotors detected | "
            "max \\|delta* error\\| (rev/s) | median z @label | median z @delta* | "
            "median mean excess (dB) |"
        )
        o.append("|---:|---:|---|---:|---:|---:|---:|---:|")
        for c in st["level_sweep"] + [st["offset_case"]]:
            zs = [float(x["z_sign"]) for x in c["per_rotor"]]
            ex = [float(x["mean_excess_db"]) for x in c["per_rotor"]]
            zd = [float(x["z_sign_delta_star"]) for x in c["per_rotor"]]
            o.append(
                f"| {c['comb_level_db']:+.0f} | {c['true_delta_rev_s']:+.1f} | {c['verdict']} | "
                f"{c['n_rotors_detected']}/{c['n_rotors']} | "
                f"{c['max_abs_delta_error_rev_s']} | {np.median(zs):.2f} | "
                f"{np.median(zd):.2f} | {np.median(ex):.2f} |"
            )
        o.append("")
        off = st["offset_case"]
        o.append(
            f"So the check finds a comb planted ON the label down to {st['sensitivity_db']} dB "
            "on every rotor (and on one rotor at -30 dB), calls it REGISTERED with `delta*` "
            f"recovered to {max(abs(float(x['delta_error_rev_s'])) for x in st['level_sweep'][0]['per_rotor']):.3f} "
            "rev/s; loses it at -36 dB WITHOUT a false MISREGISTERED "
            f"(`no_false_positive = {st['no_false_positive']}`); and calls the "
            f"+{st['offset_rev_s']} rev/s-offset comb MISREGISTERED with `delta*` recovered to "
            f"{off['max_abs_delta_error_rev_s']} rev/s, `z @delta*` "
            f"{min(float(x['z_sign_delta_star']) for x in off['per_rotor']):.1f}-"
            f"{max(float(x['z_sign_delta_star']) for x in off['per_rotor']):.1f} and `z @label` "
            f"{max(float(x['z_sign']) for x in off['per_rotor']):.2f}. The real windows show "
            "neither signature."
        )
        o.append("")
    o.append("## What it implies for the comb level")
    o.append("")
    cl = payload["comb_level"]
    bound = cl["comb_level"]
    o.append(f"* `exact re-evaluation`: {cl['exact_whittle_reevaluation']}.")
    o.append(f"* `proxy`: {cl['proxy']}.")
    o.append(
        f"* proxy recovery from re-registering at `delta*`: frame median "
        f"{cl['proxy_recovery_db']['frame_median']} dB "
        f"(max {cl['proxy_recovery_db']['frame_max']}); order-tracked median "
        f"{cl['proxy_recovery_db']['order_tracked_median']} dB "
        f"(max {cl['proxy_recovery_db']['order_tracked_max']})."
    )
    o.append(
        f"* comb level the spectra support, read off the self-test calibration: "
        f"`comb_level_db = {bound.get('comb_level_db')}`, "
        f"`upper bound = {bound.get('comb_level_upper_bound_db')}` dB re the window's own band "
        f"power, from a measured median excess of {bound.get('observed_median_excess_db')} dB "
        f"and median z of {bound.get('observed_median_z_sign')}."
    )
    o.append("")
    calls = [
        x["verdict"]
        for row in payload["windows"]
        for x in row["verdict"]["per_rotor"]
        if x["verdict"] != "UNRESOLVABLE"
    ]
    n_reg = calls.count("REGISTERED")
    n_mis = calls.count("MISREGISTERED")
    n_none = calls.count("NO_COMB")
    o.append(
        f"**Conclusion.** Over the {len(calls)} rotor reads (5 windows x 4 rotors x "
        f"{len(payload['label_keys'])} label tracks) the check returns {n_mis} MISREGISTERED, "
        f"{n_reg} REGISTERED and {n_none} NO_COMB. Nothing is misregistered: where a comb is "
        "detectable at all it sits ON the label, and where it is not, no offset in "
        "+-1.5 rev/s finds one. So the first branch of the question holds — **the likelihood's "
        f"quiet comb is honest**. Re-registering would buy "
        f"{cl['proxy_recovery_db']['order_tracked_median']} dB (median, order-tracked; "
        f"{cl['proxy_recovery_db']['frame_median']} dB on the likelihood's own frame), not "
        f"{cl['disagreement_db']} dB. The resolved comb in these windows is at most "
        f"{bound.get('comb_level_upper_bound_db')} dB of the window's own band power, so the "
        f"band-level-matched arm the gate likes (+{LEVEL_MATCHED_SHIFT_DB} dB, comb band power "
        "= the real clip's band power) is louder than ANY comb the spectrum contains by at "
        f"least {abs(float(bound.get('comb_level_upper_bound_db') or 0.0)):.0f} dB. The "
        f"{cl['disagreement_db']} dB disagreement is therefore not a registration bug and not "
        "a likelihood bug: the gate's scorer needs a comb the DREGON room-2 cruise spectrum "
        "does not have, and R3's frozen-comb `c` cannot serve both."
    )
    o.append("")
    o.append("## Figures")
    o.append("")
    for f in figures:
        o.append(f"* `{f}`")
    o.append("")
    return "\n".join(o)


# ── the run ─────────────────────────────────────────────────────────────────


def _strip(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--recordings", nargs="*", default=None)
    ap.add_argument("--keys", nargs="*", default=list(LABEL_KEYS))
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--no-selftest",
        action="store_true",
        help="skip the synthetic-comb self-test (it is what licenses a null result)",
    )
    args = ap.parse_args(argv)

    supports = list(GT.DREGON_CRUISE_SUPPORTS)
    if args.recordings:
        supports = [s for s in supports if s.recording in set(args.recordings)]
    if not supports:
        die("no DREGON cruise support selected")
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for support in supports:
        for key in args.keys:
            row = run_window(support, key)
            row["verdict"] = verdict(row)
            rows.append(row)
            path = out / f"{support.recording}__{key}.json"
            path.write_text(json.dumps(_strip(row), indent=1))
            print(
                f"{support.recording:34s} {key:15s} -> {row['verdict']['verdict']}  "
                f"({path.stat().st_size} bytes)"
            )

    diffs = []
    for support in supports:
        per_key = {r["rps_key"]: r for r in rows if r["recording"] == support.recording}
        if len(per_key) < 2:
            continue
        a, b = (per_key[k]["label"]["mean_rev_s"] for k in list(per_key)[:2])
        diffs += [abs(float(x) - float(y)) for x, y in zip(a, b, strict=True)]
    payload: dict[str, Any] = dict(
        schema=SCHEMA,
        git=git_rev(),
        supports=[s.as_dict() for s in supports],
        label_keys=list(args.keys),
        constants=dict(
            reg_half_width_rev_s=REG_HALF_WIDTH,
            reg_step_rev_s=REG_STEP,
            window_bins=WINDOW_BINS,
            drift_factor=DRIFT_FACTOR,
            window_spacing_fraction=WINDOW_SPACING_FRACTION,
            snr_cap_db=SNR_CAP_DB,
            line_snr_threshold_db=LINE_SNR_THRESHOLD_DB,
            floor_half_hz=FLOOR_HALF_HZ,
            floor_min_bins=FLOOR_MIN_BINS,
            order_track_upsample=ORDER_TRACK_UP,
            order_track_width_hz=ORDER_TRACK_WIDTH_HZ,
            verdict_analysis=VERDICT_ANALYSIS,
        ),
        label_agreement=dict(
            max_abs_mean_diff_rev_s=_r(max(diffs), 4) if diffs else None,
            n_compared=len(diffs),
        ),
        windows=[_strip(r) for r in rows],
        self_test=(None if args.no_selftest else self_test(supports[0], args.keys[0])),
    )
    payload["comb_level"] = comb_level_implication(
        [r for r in rows if r["rps_key"] == args.keys[0]], payload["self_test"]
    )
    figures: list[str] = []
    if not args.no_figures:
        figures = write_figures(rows, out)
    (out / "registration.json").write_text(json.dumps(payload, indent=1))
    (out / "findings.md").write_text(findings(payload, figures))
    print(f"wrote {out}/registration.json and findings.md; {len(figures)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
