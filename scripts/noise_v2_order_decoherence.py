"""How much of a rotor's harmonic phase noise is SHARED across orders, and how much is not?

The revised noise model ties every harmonic of one rotor to a single shaft
phase: order ``k`` carries ``k * theta(t)``. If that is the whole story, the
phase increment of every order over a lag ``tau`` is ``k`` times one number,
and a renderer needs one phase process per rotor. The prior coherence probes
(`docs/experiments/vk-decomposition.md`, `docs/experiments/stochastic-fit.md`)
said the harmonics do NOT move in lockstep even on a static single-motor
bench, but they returned correlations, not a variance. This script returns the
variance.

The decomposition, per microphone and per lag::

    dphi_k(t; tau) = k * dtheta(t; tau) + eps_k(t; tau)

``dtheta`` is the SHARED term (shaft speed error plus any propagation term that
scales with ``k``); ``eps_k`` is the INDEPENDENT per-order term - the
decoherence between harmonics that the tied model cannot represent. The script
measures ``Var[dtheta](tau)``, ``Var[eps_k](tau)``, their ratio, and whether
``eps`` is correlated across orders or across microphones.

Support: the DREGON single-motor bench, ``motor_Motor{1-4}_{70,80,90}``, 8
microphones, 44.1 kHz.

Three instrument problems had to be solved before the numbers mean anything,
and each is visible in the output:

1. **The motor does not run for the whole recording.** Eleven of the twelve
   files are 25 s long and the motor runs for about 11 s of that (roughly
   t = 4.5 s to t = 16 s); the rest is room noise. A fixed "skip 3 s, take
   24 s" window is therefore half silence, and the demodulated phase of the
   silent half is uniform noise, which alone produces a linearly growing
   phase-increment variance at EVERY order. The analysis window is detected
   from the level track (:func:`active_window`) and reported per recording.

2. **The unwrapped phase slips.** In the mandated band ``+-0.4 * rate`` the
   line-to-floor ratio of a high order on this bench is only a few dB, so the
   per-sample phase error is a large fraction of a radian and ``np.unwrap``
   takes the wrong branch a few times per hundred samples. Each slip adds
   ``(2 pi)^2`` to the increment variance and the slip count random-walks, so
   the unwrapped estimator returns a variance that grows linearly in ``tau``
   no matter what the phase does - it reported ``Var[dphi_1(500 ms)] = 207``
   rad^2 for an order whose line is 0.1 Hz wide. The headline variance is
   therefore read from the AMPLITUDE-WEIGHTED CIRCULAR mean of the de-rotated
   increment phasor, ``V = -2 log |<z(t+tau) z*(t) e^{-i k dtheta}>|``, which
   cannot slip. The unwrapped estimator is still computed and reported
   (``V_unwrap``) so the size of the artefact is on the record, and it is used
   only for the low-order shaft estimate, where the gate keeps the slip rate
   below a percent.

3. **The shaft estimate leaks into the residual.** ``dtheta_hat`` is a
   weighted sum of the same noisy increments, so its error ``e`` reappears in
   every residual as ``-k e``, i.e. as a spurious ``k^2 Var[e]`` in
   ``Var[eps_k]``. ``Var[e]`` is measured, not assumed: the shaft band is split
   into three disjoint order groups, the three independent shaft estimates are
   differenced pairwise, and ``B = Var[e]`` follows from the three pairwise
   variances. Orders inside the shaft band are de-rotated leave-group-out.

The remaining bias is the additive-noise phase floor, which is removed by a
PAIRED control: for every recording a synthetic rotor is rendered with the
same rate, the same per-(order, microphone) line-to-noise ratio and only a
shared integrated-OU shaft, and its residual - the floor - is subtracted. The
second control adds a known independent per-order Wiener phase
(``D_k = 0.1 k`` rad^2/s) to the same render and must come back out.

``--long`` answers the follow-up question the 0.5 s ladder cannot: the term it
returned grows as ``tau^q`` with ``q`` near 1, which is a Wiener phase per
order and has NO ceiling, while a per-order OU PHASE saturates after
``tau_c = 1/lambda_eps`` and is the same curve below it. The long mode runs
the identical estimator and the identical controls out to 5 s on the windows
long enough to hold three of that lag - the level-detected 31 s active run of
``motor_Motor1_70`` and every single-rotor ``noise-v2-bench-points`` point
whose published span reaches 20 s - and fits both shapes per order, reporting
``q``, ``tau_c`` and the AIC difference. Control (i) calibrates a plateau and
control (ii) a continued growth at exactly those lags.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import signal  # noqa: E402

from utils.demod import demodulate, residual_frequency  # noqa: E402

# An order whose line never clears the gate on any microphone has no cell at
# all, so the pooled mean of that column is legitimately NaN. numpy warns once
# per such column and the warnings carry no information here.
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# ═════════════════════════════════════════════════════════════════════════════
# Frozen analysis constants
# ═════════════════════════════════════════════════════════════════════════════

#: Native audio rate of the DREGON bench recordings.
FS_NATIVE = 44100.0
#: Orders demodulated.
K_MAX = 40
#: Bench throttle setpoints analysed (one recording per motor per setpoint).
SETPOINTS = (70, 80, 90)
#: Recording-id pattern of the single-motor bench split.
BENCH_RE = re.compile(r"motor_(Motor\d)_(\d+)$")

#: Audio is decimated by this factor BEFORE demodulation. Order 40 of the
#: fastest bench setpoint sits at 40 * 89.1 = 3566 Hz, so a 4x decimation
#: (Nyquist 5512 Hz) keeps every analysed line with margin and makes the
#: demodulation ladder four times cheaper. The anti-alias corner is well above
#: the highest line and its stopband is 79 dB down at the frequency that would
#: alias onto it. Verified equal to the full-rate pipeline to three decimals
#: in every SNR cell.
PRE_DECIM = 4
#: Anti-alias corner of that decimation, Hz.
AA_CORNER_HZ = 4200.0
#: Anti-alias Butterworth order (applied zero-phase, so the power response is
#: ``|H|^4``).
AA_ORDER = 8
#: Working rate after pre-decimation, Hz.
FS_WORK = FS_NATIVE / PRE_DECIM
#: Baseband decimation from the working rate. 11025 / 55 = 200.45 Hz, the
#: mandated ~200 Hz phase grid. The baseband is already limited to
#: ``+-0.4 * rate`` <= 36 Hz, far inside the 100 Hz output Nyquist, so plain
#: subsampling adds nothing.
BB_DECIM = 55
#: Phase-grid rate, Hz.
FS_PHASE = FS_WORK / BB_DECIM

#: Demodulation half-band as a fraction of the shaft rate: half the harmonic
#: spacing, the widest band that cannot see a neighbouring order.
BAND_FRAC = 0.4
#: Orders at which the constant carrier is refined with
#: :func:`utils.demod.residual_frequency`, in order. The first pass pulls the
#: high-order residual inside the band; the second centres it. Measured on
#: ``motor_Motor1_80``: the survey rate leaves the order-40 line 6.3 Hz off
#: centre, one pass at k=10 leaves 0 Hz at k=10 but 0.8 Hz at k=40, and the
#: pair (10, 40) leaves <= 0.8 Hz at every order up to 40.
REFINE_ORDERS = (10, 40)
#: Seconds trimmed from each end of the demodulation window before the carrier
#: refinement reads the residual frequency.
REFINE_TRIM_S = 0.3

#: Activity detection: level track hop, the dB drop below the 90th percentile
#: that still counts as "motor running", and the seconds trimmed from each end
#: of the detected run.
LEVEL_HOP_S = 0.05
LEVEL_DROP_DB = 6.0
LEVEL_PCT = 90.0
ACTIVE_EDGE_S = 0.5
#: Longest analysis window kept, seconds (only ``motor_Motor1_70`` is long
#: enough to reach it).
MAX_SEG_S = 24.0
#: Shortest analysis window accepted, seconds.
MIN_SEG_S = 5.0
#: Seconds trimmed from each end of the phase series (filter settling of the
#: ``+-0.4 rate`` low-pass is a few tenths of a second).
PHASE_EDGE_S = 0.5

#: Lags at which the structure functions are tabulated, milliseconds.
LAGS_MS = (5, 10, 20, 50, 100, 200, 500, 1000)
#: The two lags quoted in the summary table.
TABLE_LAGS_MS = (50, 500)
#: Lag at which the correlation matrices are read.
CORR_LAG_MS = 50

# ── the long-lag extension (``--long``, ``--max-lag-s``) ─────────────────────
#: The ladder above is frozen. ``--max-lag-s`` admits the entries of this
#: extension at or below it, so the default (0.5 s, i.e. none of them) runs
#: exactly the published analysis.
LAGS_MS_LONG = (2000, 3000, 5000)
#: Longest lag admitted when ``--long`` is given without ``--max-lag-s``.
LONG_MAX_LAG_S = 5.0
#: A lag is only tabulated when this many of it fit inside the analysis
#: window. A window holding fewer than three disjoint increments does not
#: estimate their variance, it reports one draw of it.
LAG_WINDOW_FRAC = 3.0
#: Longest analysis window kept in ``--long`` mode, seconds, i.e. no cap: the
#: published :data:`MAX_SEG_S` exists because the 0.5 s ladder needed no more,
#: and the 5 s ladder needs 15 s of window. The window is the detected active
#: run and its length is reported per support.
MAX_SEG_LONG_S = 45.0
#: Shortest published span of a single-rotor ``noise-v2-bench-points`` point
#: that enters the long support set, seconds.
LONG_SPAN_MIN_S = 20.0
#: Lags quoted in the long-lag table, milliseconds.
LONG_TABLE_LAGS_MS = (500, 1000, 2000, 5000)
#: Lag at which an order is declared to be AT the estimator ceiling, i.e. the
#: end of the published ladder.
LONG_CEIL_LAG_MS = 500
#: Shortest lag entering the long-lag shape comparison, milliseconds. Below it
#: the curve is still climbing out of the additive-noise floor, which is flat
#: in ``tau`` and would pull both shapes the same wrong way.
LONG_FIT_TAU_LO_MS = 200.0
#: Fewest curve points a shape comparison accepts (two free parameters plus a
#: variance, so four points is the smallest honest fit).
LONG_FIT_MIN_POINTS = 4
#: Growth of a curve across the fitted lags - ``V(tau_max)/V(tau_min)`` - at
#: or below which it carries no ``tau`` dependence at all. Neither shape is
#: then being chosen on evidence: the curve is a plateau already reached (or
#: an additive-noise floor, which is what control (i) returns).
FLAT_GROWTH_RATIO = 1.25
#: AIC difference that decides between the two long-lag shapes. Below it the
#: two fit equally well and the answer is "undetermined".
AIC_DECISIVE = 2.0
#: The bench recording whose level-detected active run is the longest single
#: stationary window this project holds (31 s of continuous motor).
LONG_LEVEL_RECORDING = "motor_Motor1_70"

#: In-band line gate, dB: peak of the ``|z_k|^2`` spectrum over the median of
#: the same spectrum inside the demodulation band.
SNR_MIN_DB = 10.0
#: Highest order used for the shaft estimate. Above this the per-sample phase
#: is noise-dominated and its unwrapped phase slips.
K_SHAFT = 10
#: Microphones on which an order must clear the gate before that order enters
#: the power-law fit. A per-cell gate alone selects the luckiest microphone of
#: a marginal order - its SNR estimate is high BECAUSE its own noise
#: fluctuated low - which under-states that cell's floor and over-states its
#: net residual. The weak odd orders of this two-bladed rotor clear 10 dB on
#: only 0-5 of the 8 channels, so this bar is what keeps the fit honest. Such
#: orders are still reported in the table.
ORDER_MIN_MICS = 5
#: Number of disjoint order groups the shaft band is split into, so that the
#: shaft-estimate error can be measured from the group disagreement.
N_GROUPS = 3
#: Slip censoring of the one-sample wrapped increments, in robust sigmas. The
#: true per-sample increment never exceeds half a radian on this bench, so a
#: 6-sigma censor only removes noise; it changes the shaft structure function
#: by 1 % and is reported as ``slip_frac``.
SLIP_SIGMA = 6.0

#: Ceiling of the circular estimator. ``V = -2 log |gamma|`` stops being
#: informative once ``|gamma|`` approaches its own sampling floor; cells above
#: this are reported but excluded from the fit. Validated on control (ii): the
#: planted ``D_k`` is returned to within 30 % wherever ``V_obs <= 3`` and is
#: lost above it.
V_CEIL = 3.0

#: Detection bar for a net per-order term, in standard errors of the
#: microphone mean. Below about 50 ms the true per-order term of a low order
#: is far under the additive-noise floor, so the difference of the two
#: microphone means is noise around zero; without this bar such cells enter
#: the log-log fit with random tiny positive values and dominate its residual.
DETECT_SIGMA = 2.0
#: Power-law fit domain: ``V_eps(k, tau) = a k^p tau^q``.
FIT_K_LO, FIT_K_HI = 4, 40
FIT_TAU_LO_MS, FIT_TAU_HI_MS = 10.0, 500.0
#: Bootstrap draws over recordings for the (a, p, q) confidence interval.
N_BOOT = 2000
BOOT_SEED = 20260917
#: Shortest lag at which the additive-noise floor is expected to be flat in
#: ``tau``. The noise inside a ``+-0.4 rate`` band is correlated over about
#: ``1/(2 band)`` = 16 ms, so below that the two samples of an increment share
#: part of their noise and the floor is genuinely smaller.
FLAT_LAG_LO_MS = 50.0

#: The bench shaft fit this study is compared against
#: (``results/noise_v2/shaft/findings.md``, acoustic row ``dregon_bench``).
SHAFT_SIGMA_NU = 1.77
SHAFT_LAM = 5.48
#: Planted per-order diffusion of control (ii), ``D_k = CTRL_D1 * k`` rad^2/s.
CTRL_D1 = 0.1
#: Seeds of the two controls (the render is otherwise identical, so the second
#: control is exactly the first plus the planted Wiener).
CTRL_SEED = 101
CTRL_PSI_SEED = 777
#: Amplitude-calibration passes of a control: the planted shaft broadens the
#: high-order lines and pushes part of their power out of the band, so the
#: amplitudes are rescaled until the control's measured line-to-noise ratio
#: matches the recording's.
CTRL_CAL_PASSES = 3

#: Verdict thresholds for the correlation read-outs.
CORR_INDEP_MAX = 0.10
CORR_SHARED_MIN = 0.50
#: Orders over which the cross-microphone verdict is read: above the bottom of
#: the fit range and below the order where the residual saturates the circular
#: estimator.
MID_K = slice(FIT_K_LO - 1, 24)

#: Synthetic support -> the support whose residual is its floor. ``matched``
#: and ``nominal`` are their own floor, so their ``v_leak_corrected`` IS the
#: floor and their net is identically zero; ``matched_b`` against ``matched``
#: is the null of the whole subtraction, and each ``*_planted`` row is
#: corrected with the floor of the shaft it was rendered on.
CTRL_REFERENCE = {
    "matched": "matched",
    "matched_b": "matched",
    "matched_planted": "matched",
    "nominal": "nominal",
    "nominal_planted": "nominal",
}

ORDERS = np.arange(1, K_MAX + 1)

DEFAULT_OUT = ROOT / "results" / "noise_v2" / "decoherence"
DEFAULT_OUT_LONG = ROOT / "results" / "noise_v2" / "decoherence_long"
DEFAULT_FIGS = ROOT / "docs" / "explainers" / "noise-model-v2-plan"
SURVEY_DIR = ROOT / "results" / "noise_v2" / "survey"


# ═════════════════════════════════════════════════════════════════════════════
# Model algebra
# ═════════════════════════════════════════════════════════════════════════════


def ou_structure(tau: np.ndarray | float, sigma: float, lam: float) -> np.ndarray:
    """``Var[theta(t+tau) - theta(t)]`` of an integrated Ornstein-Uhlenbeck speed error."""
    u = lam * np.asarray(tau, dtype=np.float64)
    return 2.0 * sigma**2 * (u + np.expm1(-u)) / lam**2


def phase_floor(snr: np.ndarray) -> np.ndarray:
    """Circular-estimator phase floor of additive noise, ``2 log(1 + 1/SNR)`` rad^2.

    A line of power ``P`` in additive noise of in-band power ``N`` has
    ``|gamma| = 1/(1 + N/P)`` at any lag beyond the noise correlation time, so
    the estimator reads ``2 log(1 + 1/SNR)``. In the small-noise limit this is
    ``2/SNR``, i.e. twice the ``1/(2 SNR)`` per-sample phase variance, because
    the increment differences two independent samples.
    """
    return 2.0 * np.log1p(1.0 / np.maximum(np.asarray(snr, dtype=np.float64), 1e-12))


# ═════════════════════════════════════════════════════════════════════════════
# Loading and windowing
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class Job:
    """One recording to analyse and how its analysis window is cut.

    ``window`` is a published ``(start_s, span_s)`` slice - the stationary
    window of a ``noise-v2-bench-points`` point - and when it is absent the
    window is detected from the level track (:func:`active_window`).
    """

    name: str
    audio: np.ndarray
    rate_seed: float | None = None  # rev/s from the fit-point manifest
    window: tuple[float, float] | None = None
    max_seg_s: float = MAX_SEG_S
    window_source: str = "level"
    meta: dict[str, Any] = field(default_factory=dict)


def _is_survey_setpoint(rid: str) -> bool:
    m = BENCH_RE.match(rid)
    return m is not None and int(m.group(2)) in SETPOINTS


def load_bench(limit: int | None = None) -> list[Job]:
    """Every ``motor_Motor{1-4}_{70,80,90}`` recording of the DREGON bench split."""
    return sorted(
        (
            Job(name=rid, audio=audio)
            for rid, audio in dregon_motor_audio(_is_survey_setpoint, limit)
        ),
        key=lambda j: j.name,
    )


def dregon_motor_audio(
    want: Callable[[str], bool], limit: int | None = None
) -> list[tuple[str, np.ndarray]]:
    """``(recording id, (M, T) native audio)`` of the motor-split frames ``want`` selects."""
    from data_processing.frames import meta_dict
    from data_processing.streams import iter_published_frames

    out: list[tuple[str, np.ndarray]] = []
    for frame in iter_published_frames("DREGON-frames"):
        meta = meta_dict(frame)
        if meta.get("split") != "motor":
            continue
        rid = str(meta.get("recording_id"))
        if not want(rid):
            continue
        sr = float(frame["audio"].tindex.sr)
        if sr != FS_NATIVE:
            raise RuntimeError(f"{rid} is at {sr} Hz, the ladder is frozen at {FS_NATIVE} Hz")
        out.append((rid, np.asarray(frame["audio"].data, np.float64)))
        if limit is not None and len(out) >= limit:
            break
    return out


def long_bench_points() -> list[dict[str, Any]]:
    """Single-rotor fit points whose published span reaches :data:`LONG_SPAN_MIN_S`.

    The list comes from the committed manifest that DEFINES the published
    ``noise-v2-bench-points`` dataset (``derivations.NOISE_V2_MANIFEST``), so
    the speed label, the window and the identity are the dataset's own.
    """
    from data_processing.derivations import NOISE_V2_MANIFEST

    points = json.loads(NOISE_V2_MANIFEST.read_text())["points"]
    keep = [
        p for p in points if int(p["n_rotors"]) == 1 and float(p["publish_s"]) >= LONG_SPAN_MIN_S
    ]
    return sorted(keep, key=lambda p: str(p["key"]))


def load_long(limit: int | None = None) -> list[Job]:
    """The long support set: the longest bench run plus every long single-rotor point.

    ``motor_Motor1_70`` enters twice on purpose - once on the level-detected
    active run (the longest window this bench holds) and once on its published
    30 s fit-point window - so that the long-lag shape can be read against the
    window rule as well as against the lag.

    The point audio is the source recording sliced exactly as
    :func:`data_processing.derivations._noise_v2_frame` slices it, which is
    what the published dataset holds; all long single-rotor points come from
    the DREGON motor split.
    """
    points = long_bench_points()
    foreign = sorted({str(p["source"]) for p in points} - {"DREGON-frames"})
    if foreign:
        raise RuntimeError(f"long fit points from unsupported sources: {foreign}")
    ids = {str(p["source_id"]) for p in points} | {LONG_LEVEL_RECORDING}
    audio = dict(dregon_motor_audio(lambda rid: rid in ids))
    missing = sorted(ids - set(audio))
    if missing:
        raise RuntimeError(f"DREGON-frames did not yield {missing}")

    # The manifest speed of the same recording seeds the carrier, so the long
    # run needs nothing from the gitignored survey directory (it is the same
    # estimate: the manifest speeds ARE the survey's output).
    seeds = {str(p["source_id"]): float(p["speed_rev_s"][0]) for p in points}
    jobs = [
        Job(
            name=LONG_LEVEL_RECORDING,
            audio=audio[LONG_LEVEL_RECORDING],
            rate_seed=seeds.get(LONG_LEVEL_RECORDING),
            max_seg_s=MAX_SEG_LONG_S,
            window_source="level",
            meta={"dataset": "DREGON-frames", "recording_id": LONG_LEVEL_RECORDING},
        )
    ]
    for p in points:
        jobs.append(
            Job(
                name=str(p["key"]),
                audio=audio[str(p["source_id"])],
                rate_seed=float(p["speed_rev_s"][0]),
                window=(float(p["publish_start_s"]), float(p["publish_s"])),
                window_source="published",
                meta={
                    "dataset": "noise-v2-bench-points",
                    "recording_id": str(p["source_id"]),
                    "throttle": p.get("throttle"),
                    "publish_start_s": float(p["publish_start_s"]),
                    "publish_s": float(p["publish_s"]),
                    "speed_rev_s": [float(v) for v in p["speed_rev_s"]],
                    "speed_tolerance_rev_s": float(p["speed_tolerance_rev_s"]),
                },
            )
        )
    return jobs if limit is None else jobs[:limit]


def survey_rates() -> dict[str, float]:
    """Per-recording shaft rate of the corpus survey, rev/s."""
    rates: dict[str, float] = {}
    for path in sorted(SURVEY_DIR.glob("bench_speeds_*.json")):
        blob = json.loads(path.read_text())
        for row in blob.get("rows", []):
            rid = str(row.get("id", ""))
            speeds = row.get("speed_rev_s") or []
            if BENCH_RE.match(rid) and speeds:
                rates.setdefault(rid, float(speeds[0]))
    return rates


def active_window(audio: np.ndarray, fs: float = FS_NATIVE) -> tuple[int, int]:
    """Longest run of samples where the motor is running, as ``[start, stop)``.

    The bench files start and end with room noise; the run is 30-40 dB above
    it, so the longest stretch of the level track within
    :data:`LEVEL_DROP_DB` of its 90th percentile is the run. The percentile
    rather than the maximum, because the start-up transient of some files
    peaks above the plateau.
    """
    hop = int(LEVEL_HOP_S * fs)
    n = audio.shape[-1] // hop
    if n < 4:
        return 0, audio.shape[-1]
    frames = audio[..., : n * hop].reshape(audio.shape[0], n, hop)
    level = 10.0 * np.log10(np.maximum(np.mean(frames**2, axis=(0, 2)), 1e-30))
    good = level >= np.percentile(level, LEVEL_PCT) - LEVEL_DROP_DB
    best = (0, 0)
    i = 0
    while i < n:
        if good[i]:
            j = i
            while j < n and good[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    return best[0] * hop, best[1] * hop


def analysis_segment(
    audio: np.ndarray, max_seg_s: float = MAX_SEG_S
) -> tuple[np.ndarray, float, float]:
    """``(mean-removed segment, start seconds, length seconds)`` of the stationary run."""
    a, b = active_window(audio)
    a += int(ACTIVE_EDGE_S * FS_NATIVE)
    b -= int(ACTIVE_EDGE_S * FS_NATIVE)
    b = min(b, a + int(max_seg_s * FS_NATIVE))
    if b - a < MIN_SEG_S * FS_NATIVE:
        raise RuntimeError(f"stationary run {(b - a) / FS_NATIVE:.2f} s below {MIN_SEG_S} s")
    seg = audio[:, a:b]
    return seg - seg.mean(axis=1, keepdims=True), a / FS_NATIVE, (b - a) / FS_NATIVE


def published_segment(
    audio: np.ndarray, start_s: float, span_s: float
) -> tuple[np.ndarray, float, float]:
    """``(mean-removed segment, start seconds, length seconds)`` of a published window.

    The slice is the one ``derivations._noise_v2_frame`` publishes for that
    fit point, so the segment analysed here is the dataset's own sample.
    """
    start = int(round(start_s * FS_NATIVE))
    stop = min(audio.shape[-1], start + int(round(span_s * FS_NATIVE)))
    start = max(0, min(start, max(0, stop - 1)))
    if stop - start < MIN_SEG_S * FS_NATIVE:
        raise RuntimeError(
            f"published window {(stop - start) / FS_NATIVE:.2f} s below {MIN_SEG_S} s"
        )
    seg = audio[:, start:stop]
    return (
        seg - seg.mean(axis=1, keepdims=True),
        start / FS_NATIVE,
        (stop - start) / FS_NATIVE,
    )


def job_segment(job: Job) -> tuple[np.ndarray, float, float]:
    """The analysis window of one job, published slice or detected active run."""
    if job.window is None:
        return analysis_segment(job.audio, job.max_seg_s)
    return published_segment(job.audio, *job.window)


def pre_decimate(seg: np.ndarray) -> np.ndarray:
    """Anti-alias and decimate by :data:`PRE_DECIM` onto the working rate."""
    sos = signal.butter(AA_ORDER, AA_CORNER_HZ, btype="low", fs=FS_NATIVE, output="sos")
    return np.ascontiguousarray(signal.sosfiltfilt(sos, seg, axis=-1)[..., ::PRE_DECIM])


# ═════════════════════════════════════════════════════════════════════════════
# Demodulation ladder, carrier refinement, line SNR
# ═════════════════════════════════════════════════════════════════════════════


def refine_rate(work: np.ndarray, rate: float, order: int) -> tuple[float, float]:
    """One carrier-refinement pass at ``order``: ``(new rate, residual Hz)``.

    A constant rate error ``d`` puts the order-``k`` line at ``k d`` Hz inside
    the band, so the power-weighted residual frequency of the order-``k``
    baseband divided by ``k`` is the rate error.
    """
    z = demodulate(work, np.full(work.shape[-1], rate), BAND_FRAC * rate, FS_WORK, order=order)
    trim = int(REFINE_TRIM_S * FS_WORK)
    z = z[:, trim:-trim]
    freq = residual_frequency(z, FS_WORK)
    weight = np.abs(z) ** 2
    df = float(np.sum(freq * weight) / np.sum(weight))
    return rate + df / order, df


def demod_ladder(work: np.ndarray, rate: float) -> np.ndarray:
    """``(K, M, T)`` complex baseband of orders ``1..K_MAX`` on the phase grid."""
    band = BAND_FRAC * rate
    carrier = np.full(work.shape[-1], rate)
    return np.stack(
        [
            demodulate(work, carrier, band, FS_WORK, order=k)[:, ::BB_DECIM]
            for k in range(1, K_MAX + 1)
        ]
    )


def line_snr(z: np.ndarray, band: float) -> tuple[np.ndarray, np.ndarray]:
    """``(peak/median, line power / noise power)`` per ``(order, microphone)``.

    The first is the gate and the weight the method asks for: the peak of the
    ``|z_k|^2`` spectrum over the median of the same spectrum inside the band.
    The second is the ratio that sets the phase floor - total in-band power
    minus the floor, over the floor - and is what the controls are rendered
    to match.
    """
    n = z.shape[-1]
    nper = min(1024, n)
    freq, spec = signal.welch(
        z,
        fs=FS_PHASE,
        nperseg=nper,
        noverlap=nper // 2,
        return_onesided=False,
        detrend=False,  # pyright: ignore[reportArgumentType]
        axis=-1,
    )
    freq = np.fft.fftshift(freq)
    spec = np.fft.fftshift(spec, axes=-1)
    inband = np.abs(freq) <= band
    sb = spec[..., inband]
    peak = sb.max(axis=-1)
    median = np.maximum(np.median(sb, axis=-1), 1e-300)
    # The Butterworth skirt lowers the floor near the band edges, so the noise
    # level is read where the floor is flat: outside the line, inside the
    # passband.
    flank = (np.abs(freq) > 0.3 * band) & (np.abs(freq) < 0.8 * band)
    floor = np.maximum(np.median(spec[..., flank], axis=-1), 1e-300)
    p_noise = floor * 2.0 * band
    p_total = np.mean(np.abs(z) ** 2, axis=-1)
    p_line = np.maximum(p_total - p_noise, 1e-30)
    return peak / median, p_line / p_noise


# ═════════════════════════════════════════════════════════════════════════════
# Phase, shaft estimate, residual
# ═════════════════════════════════════════════════════════════════════════════


def unwrap_censored(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(unwrapped phase, slip fraction)`` with cycle slips censored.

    The unwrapped phase is the cumulative sum of the one-sample WRAPPED
    increments. A cycle slip is a single increment far outside the robust
    scale of the bulk; on this bench the true per-sample increment never
    leaves half a radian, so an increment past :data:`SLIP_SIGMA` robust sigmas
    is noise and is replaced by the median increment. This does not rescue the
    unwrapped estimator at low SNR - the slips that matter are inside the
    censor - but it keeps the low-order shaft estimate clean.
    """
    ang = np.angle(z)
    d = (np.diff(ang, axis=-1) + np.pi) % (2.0 * np.pi) - np.pi
    med = np.median(d, axis=-1, keepdims=True)
    scale = 1.4826 * np.median(np.abs(d - med), axis=-1, keepdims=True)
    bad = np.abs(d - med) > SLIP_SIGMA * np.maximum(scale, 1e-12)
    frac = bad.mean(axis=-1)
    d = np.where(bad, np.broadcast_to(med, d.shape), d)
    head = np.zeros(d.shape[:-1] + (1,), dtype=np.float64)
    return np.concatenate([head, np.cumsum(d, axis=-1)], axis=-1), frac


def weighted_shaft(phi: np.ndarray, weight: np.ndarray, use: np.ndarray) -> np.ndarray:
    """``(M, T)`` SNR-weighted least-squares shaft phase over the selected orders."""
    w = np.where(use, weight, 0.0)
    den = np.einsum("km,k->m", w, ORDERS**2)
    num = np.einsum("km,k,kmt->mt", w, ORDERS, phi)
    if np.any(den <= 0):
        raise RuntimeError("no usable order for the shaft estimate on some microphone")
    return num / den[:, None]


def shaft_group_phases(
    phi: np.ndarray, weight: np.ndarray, use: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``((G, M, T) group shaft phases, (K,) group index of each order)``."""
    group = (ORDERS - 1) % N_GROUPS
    out = []
    for g in range(N_GROUPS):
        sel = use & (group[:, None] == g)
        out.append(weighted_shaft(phi, weight, sel))
    return np.stack(out), group


def group_leakage(
    shaft: np.ndarray, lag: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(B_g, B_comb, dtheta_comb, group increments)`` from group disagreement.

    With three disjoint order groups the pairwise increment-difference
    variances are ``d_ij = B_i + B_j``, so ``B_i = (d_ij + d_ik - d_jk)/2``.
    The groups are then combined inverse-variance, which is both the minimum
    variance estimate of ``dtheta`` and the one whose own error
    ``B = 1/sum(1/B_g)`` is known.
    """
    d = shaft[:, :, lag:] - shaft[:, :, :-lag]
    ng, n_mic, _ = d.shape
    pair = np.zeros((ng, ng, n_mic))
    for i in range(ng):
        for j in range(i + 1, ng):
            pair[i, j] = pair[j, i] = np.var(d[i] - d[j], axis=-1)
    b = np.zeros((ng, n_mic))
    for i in range(ng):
        o = [j for j in range(ng) if j != i]
        b[i] = 0.5 * (pair[i, o[0]] + pair[i, o[1]] - pair[o[0], o[1]])
    b = np.maximum(b, 1e-12)
    inv = 1.0 / b
    b_comb = 1.0 / inv.sum(axis=0)
    dtheta = (d * inv[:, :, None]).sum(axis=0) * b_comb[:, None]
    return b, b_comb, dtheta, d


def circular_residual(z: np.ndarray, dtheta: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """``(V_eps (K, M), wrapped eps (K, M, T-lag))`` of the de-rotated increment.

    ``V = -2 log |gamma|`` of the amplitude-weighted circular mean of
    ``z_k(t+tau) z_k*(t) exp(-i k dtheta(t; tau))``. Amplitude weighting is
    what makes it slip-proof: a sample whose phasor has collapsed into the
    noise contributes little instead of contributing a random turn.
    """
    prod = (
        z[..., lag:] * np.conj(z[..., :-lag]) * np.exp(-1j * ORDERS[:, None, None] * dtheta[None])
    )
    num = np.abs(prod.sum(axis=-1))
    den = np.maximum(np.abs(prod).sum(axis=-1), 1e-300)
    return -2.0 * np.log(np.clip(num / den, 1e-8, 1.0)), np.angle(prod)


@dataclass
class LagResult:
    tau: float
    lag: int
    v_obs: np.ndarray  # (K, M) leave-group-out residual variance, rad^2
    v_unwrap: np.ndarray  # (K, M) the same from the unwrapped phase
    leak: np.ndarray  # (K, M) shaft-estimate leakage k^2 B, rad^2
    v_shaft_raw: np.ndarray  # (M,) Var[dtheta_hat]
    v_shaft: np.ndarray  # (M,) the same minus B
    b_comb: np.ndarray  # (M,) Var of the shaft-estimate error
    v_all_orders: np.ndarray  # (K, M) residual with dtheta from ALL gated orders
    v_shaft_all: np.ndarray  # (M,) Var[dtheta_hat] from all gated orders
    eps_a: np.ndarray | None = None  # wrapped eps, shaft estimate A
    eps_b: np.ndarray | None = None  # wrapped eps, shaft estimate B (independent)


@dataclass
class Support:
    name: str
    rate: float
    band: float
    keep: np.ndarray  # (K, M) gate
    snr_pm: np.ndarray  # (K, M) peak/median, linear
    snr_eff: np.ndarray  # (K, M) line/noise power, linear
    slip: np.ndarray  # (K, M) censored fraction of one-sample increments
    n_frames: int
    lags: dict[int, LagResult] = field(default_factory=dict)
    seg_start_s: float = 0.0
    seg_len_s: float = 0.0
    rate_survey: float | None = None


def analyse_support(
    name: str, z: np.ndarray, rate: float, band: float, lags_ms: tuple[int, ...] = LAGS_MS
) -> Support:
    """The whole decomposition for one demodulated recording."""
    snr_pm, snr_eff = line_snr(z, band)
    keep = 10.0 * np.log10(snr_pm) >= SNR_MIN_DB
    # The shaft band is NOT gated. The SNR weight already reduces a collapsed
    # cell to nothing (order 2 weighs 500 against 1 for a cell at 0 dB), while
    # gating it can empty one of the three order groups on one microphone and
    # leave the shaft-estimate error unmeasurable there.
    low = np.broadcast_to(ORDERS[:, None] <= K_SHAFT, keep.shape)
    phi, slip = unwrap_censored(z)
    edge = int(round(PHASE_EDGE_S * FS_PHASE))
    phi = phi[..., edge:-edge]
    zt = z[..., edge:-edge]

    shaft, group = shaft_group_phases(phi, snr_eff, low)
    shaft_all = weighted_shaft(phi, snr_eff, keep)

    sup = Support(
        name=name,
        rate=rate,
        band=band,
        keep=keep,
        snr_pm=snr_pm,
        snr_eff=snr_eff,
        slip=slip,
        n_frames=int(zt.shape[-1]),
    )
    for ms in lags_ms:
        lag = int(round(ms * 1e-3 * FS_PHASE))
        # A lag needs LAG_WINDOW_FRAC of itself inside the window: the
        # circular mean over fewer than that many disjoint increments is one
        # draw of the variance, not an estimate of it. Every published lag
        # clears this on every published window, so the rule only ever bites
        # in --long mode.
        if lag < 1 or lag >= zt.shape[-1] - 20 or LAG_WINDOW_FRAC * lag > zt.shape[-1]:
            continue
        b_g, b_comb, dtheta, d_g = group_leakage(shaft, lag)
        v_obs, eps = circular_residual(zt, dtheta, lag)
        leak = np.broadcast_to(b_comb, (K_MAX,) + b_comb.shape).copy()
        # Leave-group-out inside the shaft band, so that no order is
        # de-rotated with a shaft estimate that used it.
        inv = 1.0 / b_g
        eps_pair: list[np.ndarray] = []
        for g in range(N_GROUPS):
            rows = np.where((group == g) & (ORDERS <= K_SHAFT))[0]
            others = [j for j in range(N_GROUPS) if j != g]
            inv_o = inv[others]
            b_o = 1.0 / inv_o.sum(axis=0)
            dtheta_o = (d_g[others] * inv_o[:, :, None]).sum(axis=0) * b_o[:, None]
            v_o, eps_o = circular_residual(zt, dtheta_o, lag)
            if rows.size:
                v_obs[rows] = v_o[rows]
                leak[rows] = b_o
                eps[rows] = eps_o[rows]
            if g < 2:
                eps_pair.append(eps_o)
        # Unwrapped-phase variant of the same residual, for the record.
        d_phi = phi[..., lag:] - phi[..., :-lag]
        resid = d_phi - ORDERS[:, None, None] * dtheta[None]
        v_unwrap = resid.var(axis=-1)
        # All-gated-orders shaft estimate (the literal method), uncorrected.
        d_all = shaft_all[:, lag:] - shaft_all[:, :-lag]
        v_all, _ = circular_residual(zt, d_all, lag)

        res = LagResult(
            tau=lag / FS_PHASE,
            lag=lag,
            v_obs=v_obs,
            v_unwrap=v_unwrap,
            leak=leak * ORDERS[:, None] ** 2,
            v_shaft_raw=np.var(dtheta, axis=-1),
            v_shaft=np.maximum(np.var(dtheta, axis=-1) - b_comb, 0.0),
            b_comb=b_comb,
            v_all_orders=v_all,
            v_shaft_all=np.var(d_all, axis=-1),
        )
        if ms == CORR_LAG_MS and len(eps_pair) == 2:
            res.eps_a, res.eps_b = eps_pair[0], eps_pair[1]
        sup.lags[ms] = res
    return sup


# ═════════════════════════════════════════════════════════════════════════════
# Controls
# ═════════════════════════════════════════════════════════════════════════════


def ou_path(n: int, fs: float, sigma: float, lam: float, rng: np.random.Generator) -> np.ndarray:
    """Shaft phase of an integrated OU speed error, rad, on the ``fs`` grid."""
    dt = 1.0 / fs
    a = math.exp(-lam * dt)
    xi = rng.standard_normal(n) * (sigma * math.sqrt(1.0 - a * a))
    nu0 = rng.standard_normal() * sigma
    nu = signal.lfilter([1.0], [1.0, -a], xi, zi=[a * nu0])[0]
    return np.cumsum(nu) * dt


def unit_noise_power(rate: float, n: int) -> float:
    """In-band baseband power of unit-variance white audio, through this pipeline."""
    rng = np.random.default_rng(4242)
    z = demodulate(
        rng.standard_normal((2, n)), np.full(n, rate), BAND_FRAC * rate, FS_WORK, order=20
    )[:, ::BB_DECIM]
    return float(np.mean(np.abs(z) ** 2))


def render_control(
    rate: float,
    n: int,
    amp: np.ndarray,
    *,
    seed: int,
    psi: np.ndarray,
    theta: np.ndarray,
    d1: float = 0.0,
) -> np.ndarray:
    """``(M, n)`` synthetic rotor: shared shaft ``theta``, optional per-order Wiener, white floor."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS_WORK
    base = 2.0 * np.pi * rate * t + theta
    x = rng.standard_normal((amp.shape[1], n))
    for k in range(1, amp.shape[0] + 1):
        phase = k * base
        if d1 > 0.0:
            step = rng.standard_normal(n) * math.sqrt(2.0 * d1 * k / FS_WORK)
            phase = phase + np.cumsum(step)
        x += amp[k - 1][:, None] * np.cos(phase[None, :] + psi[k - 1][:, None])
    return x


def matched_sigma(sup: Support) -> float:
    """``sigma_nu`` of an integrated OU that reproduces this recording's own ``V_shared``.

    The mandated control (i) carries the fitted bench shaft of
    ``results/noise_v2/shaft/findings.md``, ``sigma_nu = 1.77`` rad/s. That fit
    absorbed the per-order decoherence into the shared term, so it is about
    three times the shared phase noise actually measured here - and a control
    whose shaft is three times too large is NOT a valid floor to subtract: its
    low-order lines are broader, its unwrapped low-order phase slips more, and
    its shaft-estimate error ``B`` comes out several times larger than the
    recording's. The floor that IS subtracted therefore uses the shaft this
    recording actually has, with ``lambda`` held at the fitted value.
    """
    ratios = []
    for ms, res in sup.lags.items():
        if not (FLAT_LAG_LO_MS <= ms <= FIT_TAU_HI_MS):
            continue
        shape = float(ou_structure(res.tau, 1.0, SHAFT_LAM))
        if shape > 0:
            ratios.append(float(np.mean(res.v_shaft)) / shape)
    if not ratios:
        return SHAFT_SIGMA_NU
    return float(math.sqrt(max(float(np.median(ratios)), 1e-6)))


def calibrated_render(
    sup: Support, n: int, theta: np.ndarray, psi: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """``(demodulated ladder, amplitudes, 95th-percentile dB mismatch on gated cells)``.

    The planted shaft spreads part of a high order's power outside the
    demodulation band, so the amplitude that produces a target IN-BAND
    line-to-noise ratio is not the analytic one and has to be iterated. The
    mismatch reported is the one of the ladder returned, not of an earlier
    pass, and it is read on the GATED cells: an order that never clears the
    gate takes part in nothing.
    """
    target = sup.snr_eff
    amp = 2.0 * np.sqrt(np.maximum(target, 1e-4) * unit_noise_power(sup.rate, n))
    for i in range(CTRL_CAL_PASSES + 1):
        x = render_control(sup.rate, n, amp, seed=seed, psi=psi, theta=theta)
        z = demod_ladder(x, sup.rate)
        _, eff = line_snr(z, sup.band)
        err = np.abs(10.0 * np.log10(np.maximum(eff, 1e-9) / np.maximum(target, 1e-9)))
        err_db = float(np.percentile(err[sup.keep], 95)) if sup.keep.any() else float("nan")
        if i == CTRL_CAL_PASSES:
            return z, amp, err_db
        amp = amp * np.sqrt(np.clip(target / np.maximum(eff, 1e-6), 0.05, 20.0))
    raise AssertionError("unreachable")


def build_controls(
    sup: Support, n: int, lags_ms: tuple[int, ...] = LAGS_MS
) -> tuple[dict[str, Support], dict[str, Any]]:
    """The five synthetic supports of one recording.

    ``matched``
        Shared integrated-OU shaft at the ``sigma_nu`` this recording actually
        has, plus a white floor at its measured per-cell line-to-noise ratio.
        This is the floor SUBTRACTED from the data, and it is control (i) in
        the regime the data lives in.
    ``matched_b``
        The same, with different noise, blade phases and shaft realisation. Its
        net value against ``matched`` is what the whole subtraction returns when
        there is nothing to find - the null.
    ``matched_planted``
        ``matched`` plus an independent per-order Wiener, ``D_k = CTRL_D1 * k``.
        This is control (ii) in the regime of the data, and the recovery quoted.
    ``nominal`` / ``nominal_planted``
        The two controls with the literally mandated shaft,
        ``sigma_nu = SHAFT_SIGMA_NU``, i.e. the published acoustic bench fit.
    """
    rate, band = sup.rate, sup.band
    n_mic = sup.keep.shape[1]
    psi = np.random.default_rng(CTRL_PSI_SEED).uniform(0.0, 2.0 * np.pi, (K_MAX, n_mic))
    psi_b = np.random.default_rng(CTRL_PSI_SEED + 1).uniform(0.0, 2.0 * np.pi, (K_MAX, n_mic))
    sigma = matched_sigma(sup)
    th_m = ou_path(n, FS_WORK, sigma, SHAFT_LAM, np.random.default_rng(CTRL_SEED + 1))
    th_b = ou_path(n, FS_WORK, sigma, SHAFT_LAM, np.random.default_rng(CTRL_SEED + 3))
    th_n = ou_path(n, FS_WORK, SHAFT_SIGMA_NU, SHAFT_LAM, np.random.default_rng(CTRL_SEED + 2))
    z_m, amp_m, err_m = calibrated_render(sup, n, th_m, psi, seed=CTRL_SEED)
    z_n, amp_n, err_n = calibrated_render(sup, n, th_n, psi, seed=CTRL_SEED)
    extra = {
        "matched_b": render_control(rate, n, amp_m, seed=CTRL_SEED + 50, psi=psi_b, theta=th_b),
        "matched_planted": render_control(
            rate, n, amp_m, seed=CTRL_SEED, psi=psi, theta=th_m, d1=CTRL_D1
        ),
        "nominal_planted": render_control(
            rate, n, amp_n, seed=CTRL_SEED, psi=psi, theta=th_n, d1=CTRL_D1
        ),
    }
    sups = {
        "matched": analyse_support(f"{sup.name}|ctrl_matched", z_m, rate, band, lags_ms),
        "nominal": analyse_support(f"{sup.name}|ctrl_nominal", z_n, rate, band, lags_ms),
    }
    for name, x in extra.items():
        sups[name] = analyse_support(
            f"{sup.name}|ctrl_{name}", demod_ladder(x, rate), rate, band, lags_ms
        )
    prov = {
        "sigma_matched": sigma,
        "sigma_nominal": SHAFT_SIGMA_NU,
        "lam": SHAFT_LAM,
        "snr_match_p95_db_matched": err_m,
        "snr_match_p95_db_nominal": err_n,
    }
    return sups, prov


# ═════════════════════════════════════════════════════════════════════════════
# Per-recording reduction
# ═════════════════════════════════════════════════════════════════════════════


def cell_mean(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Microphone mean of a ``(K, M)`` quantity over gated cells; NaN where none."""
    out = np.full(values.shape[0], np.nan)
    for k in range(values.shape[0]):
        m = keep[k]
        if m.any():
            out[k] = float(np.mean(values[k][m]))
    return out


def cell_sem(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Standard error of the microphone mean of a ``(K, M)`` quantity."""
    out = np.full(values.shape[0], np.nan)
    for k in range(values.shape[0]):
        m = keep[k]
        n = int(m.sum())
        if n >= 2:
            out[k] = float(np.std(values[k][m], ddof=1) / math.sqrt(n))
    return out


def reduce_support(sup: Support, floor: Support | None) -> dict[str, Any]:
    """Per-order structure functions of one recording, floor- and leakage-corrected."""
    rows: dict[str, Any] = {}
    for ms, res in sup.lags.items():
        keep = sup.keep
        v_obs = cell_mean(res.v_obs, keep)
        leak = cell_mean(res.leak, keep)
        v_lk = v_obs - leak
        entry: dict[str, Any] = {
            "tau_s": res.tau,
            "v_obs": v_obs.tolist(),
            "leakage": leak.tolist(),
            "v_leak_corrected": v_lk.tolist(),
            "v_unwrap": cell_mean(res.v_unwrap, keep).tolist(),
            "v_all_orders": cell_mean(res.v_all_orders, keep).tolist(),
            "v_shaft_raw": float(np.mean(res.v_shaft_raw)),
            "v_shaft": float(np.mean(res.v_shaft)),
            "v_shaft_all_orders": float(np.mean(res.v_shaft_all)),
            "shaft_leakage": float(np.mean(res.b_comb)),
            "n_cells": keep.sum(axis=1).tolist(),
        }
        if floor is not None and ms in floor.lags:
            fr = floor.lags[ms]
            fk = floor.keep
            f_obs = cell_mean(fr.v_obs, fk)
            f_leak = cell_mean(fr.leak, fk)
            f_net = f_obs - f_leak
            net = v_lk - f_net
            # A cell counts as a DETECTION only when the net term stands clear
            # of the error on the difference of the two microphone means. At
            # short lags the true per-order term is far below the floor, and
            # without this bar those cells enter the log-log fit with
            # essentially random tiny positive values.
            err = np.sqrt(cell_sem(res.v_obs, keep) ** 2 + cell_sem(fr.v_obs, fk) ** 2)
            sigmas = net / np.maximum(err, 1e-12)
            established = keep.sum(axis=1) >= ORDER_MIN_MICS
            resolvable = (
                established
                & np.isfinite(v_obs)
                & (v_obs <= V_CEIL)
                & np.isfinite(net)
                & (net > 0.0)
                & (sigmas >= DETECT_SIGMA)
            )
            entry.update(
                {
                    "v_floor": f_net.tolist(),
                    "v_floor_analytic": cell_mean(phase_floor(sup.snr_eff), keep).tolist(),
                    "v_net": net.tolist(),
                    "net_err": err.tolist(),
                    "detect_sigmas": sigmas.tolist(),
                    "resolvable": resolvable.tolist(),
                }
            )
        rows[str(ms)] = entry
    return rows


def correlation_reads(sup: Support) -> dict[str, Any]:
    """Cross-order and cross-microphone correlation of the residual at the corr lag."""
    res = sup.lags.get(CORR_LAG_MS)
    if res is None or res.eps_a is None or res.eps_b is None:
        return {}
    a, b = res.eps_a, res.eps_b
    keep = sup.keep
    usable = keep & (res.v_obs <= V_CEIL)

    # Cross-order: row from shaft estimate A, column from the independent B, so
    # the shared shaft-estimate error cannot manufacture a correlation.
    n_mic = a.shape[1]
    acc = np.zeros((K_MAX, K_MAX))
    cnt = np.zeros((K_MAX, K_MAX))
    for m in range(n_mic):
        ok = usable[:, m]
        if ok.sum() < 3:
            continue
        xa = a[:, m, :] - a[:, m, :].mean(axis=-1, keepdims=True)
        xb = b[:, m, :] - b[:, m, :].mean(axis=-1, keepdims=True)
        na = np.sqrt((xa**2).sum(axis=-1))
        nb = np.sqrt((xb**2).sum(axis=-1))
        c = (xa @ xb.T) / np.maximum(np.outer(na, nb), 1e-300)
        c = 0.5 * (c + c.T)
        sel = np.outer(ok, ok)
        acc[sel] += c[sel]
        cnt[sel] += 1.0
    order_corr = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)

    # Cross-microphone at the same order, mic i from A and mic j from B.
    mic_rho = np.full(K_MAX, np.nan)
    for k in range(K_MAX):
        ok = np.where(usable[k])[0]
        if ok.size < 2:
            continue
        xa = a[k, ok, :] - a[k, ok, :].mean(axis=-1, keepdims=True)
        xb = b[k, ok, :] - b[k, ok, :].mean(axis=-1, keepdims=True)
        na = np.sqrt((xa**2).sum(axis=-1))
        nb = np.sqrt((xb**2).sum(axis=-1))
        c = (xa @ xb.T) / np.maximum(np.outer(na, nb), 1e-300)
        off = ~np.eye(ok.size, dtype=bool)
        mic_rho[k] = float(np.mean(0.5 * (c + c.T)[off]))
    return {
        "lag_ms": CORR_LAG_MS,
        "order_corr": order_corr.tolist(),
        "order_usable": usable.any(axis=1).tolist(),
        "mic_rho": mic_rho.tolist(),
    }


def block_reads(order_corr: np.ndarray, usable: np.ndarray) -> dict[str, float]:
    """Off-diagonal, same-parity and cross-parity means of a cross-order matrix."""
    k = np.arange(1, order_corr.shape[0] + 1)
    ok = np.isfinite(order_corr) & np.outer(usable, usable)
    off = ok & ~np.eye(order_corr.shape[0], dtype=bool)
    odd = (k % 2) == 1
    same = off & (np.outer(odd, odd) | np.outer(~odd, ~odd))
    cross = off & ~same
    neighbour = off & (np.abs(k[:, None] - k[None, :]) == 2)

    def mean(mask: np.ndarray) -> float:
        return float(np.mean(order_corr[mask])) if mask.any() else float("nan")

    return {
        "off_diagonal": mean(off),
        "off_diagonal_abs": float(np.mean(np.abs(order_corr[off]))) if off.any() else float("nan"),
        "same_parity": mean(same),
        "cross_parity": mean(cross),
        "neighbour_dk2": mean(neighbour),
        "n_pairs": int(off.sum() // 2),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Pooling and the power-law fit
# ═════════════════════════════════════════════════════════════════════════════


def stack_field(rows: list[dict[str, Any]], ms: int, key: str) -> np.ndarray:
    """``(n_recordings, K)`` of one per-order field at one lag."""
    out = []
    for r in rows:
        e = r["orders"].get(str(ms))
        out.append(
            np.asarray(e[key], dtype=np.float64) if e and key in e else np.full(K_MAX, np.nan)
        )
    return np.stack(out)


def pooled_curves(rows: list[dict[str, Any]], lags_ms: tuple[int, ...] = LAGS_MS) -> dict[str, Any]:
    """Recording-mean structure functions, per lag."""
    out: dict[str, Any] = {}
    for ms in lags_ms:
        entries = [r["orders"].get(str(ms)) for r in rows]
        entries = [e for e in entries if e]
        if not entries:
            continue
        with np.errstate(invalid="ignore"):
            blob = {
                "tau_s": float(np.mean([e["tau_s"] for e in entries])),
                "v_shaft": float(np.nanmean([e["v_shaft"] for e in entries])),
                "v_shaft_raw": float(np.nanmean([e["v_shaft_raw"] for e in entries])),
                "v_shaft_all_orders": float(np.nanmean([e["v_shaft_all_orders"] for e in entries])),
                "shaft_leakage": float(np.nanmean([e["shaft_leakage"] for e in entries])),
                "v_obs": np.nanmean(stack_field(rows, ms, "v_obs"), axis=0).tolist(),
                "v_leak_corrected": np.nanmean(
                    stack_field(rows, ms, "v_leak_corrected"), axis=0
                ).tolist(),
                "v_unwrap": np.nanmean(stack_field(rows, ms, "v_unwrap"), axis=0).tolist(),
                "v_all_orders": np.nanmean(stack_field(rows, ms, "v_all_orders"), axis=0).tolist(),
                "v_floor": np.nanmean(stack_field(rows, ms, "v_floor"), axis=0).tolist(),
                "v_net": np.nanmean(stack_field(rows, ms, "v_net"), axis=0).tolist(),
                "n_resolvable": np.nansum(
                    stack_field(rows, ms, "resolvable").astype(bool), axis=0
                ).tolist(),
                "resolved": consensus(rows, ms).tolist(),
                "n_recordings": len(rows),
            }
        out[str(ms)] = blob
    return out


def fit_power_law(
    v: dict[int, np.ndarray],
    resolvable: dict[int, np.ndarray],
    parity: str = "all",
) -> dict[str, float] | None:
    """``log V = log a + p log k + q log tau`` by ordinary least squares.

    ``parity`` restricts the fit to ``"even"`` or ``"odd"`` orders. The two
    families of a two-bladed rotor carry very different amounts of per-order
    phase noise, so the single fit over both is a compromise between them and
    its residual is dominated by the split rather than by scatter.
    """
    rows, rhs = [], []
    for ms, vals in v.items():
        if not (FIT_TAU_LO_MS <= ms <= FIT_TAU_HI_MS):
            continue
        tau = ms * 1e-3
        for idx in range(K_MAX):
            k = idx + 1
            if not (FIT_K_LO <= k <= FIT_K_HI):
                continue
            if parity == "even" and k % 2:
                continue
            if parity == "odd" and not k % 2:
                continue
            val = vals[idx]
            if not np.isfinite(val) or val <= 0 or not resolvable[ms][idx]:
                continue
            rows.append([1.0, math.log(k), math.log(tau)])
            rhs.append(math.log(val))
    if len(rows) < 6:
        return None
    a, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    resid = np.asarray(rhs) - np.asarray(rows) @ a
    return {
        "a": float(math.exp(a[0])),
        "p": float(a[1]),
        "q": float(a[2]),
        "n_cells": len(rows),
        "rms_log_resid": float(np.sqrt(np.mean(resid**2))),
    }


def consensus(rows: list[dict[str, Any]], ms: int) -> np.ndarray:
    """``(K,)`` mask of orders whose POOLED net term is a detection at one lag.

    The per-recording flag in the JSON tests one 10 s window against its own
    microphone scatter, and a "detected in any recording" rule then fires on a
    quarter of the null control's cells. The pooled test is the right one: the
    quantity fitted is the mean over recordings, so its error is the standard
    error ACROSS recordings, and an order counts when that mean stands
    :data:`DETECT_SIGMA` such errors clear of zero. It also has to be a line -
    gated on at least :data:`ORDER_MIN_MICS` microphones in most recordings -
    and to sit under the estimator ceiling. On the null control this returns
    nothing; on the bench it returns the cells quoted.
    """
    net = stack_field(rows, ms, "v_net")
    obs = np.nanmean(stack_field(rows, ms, "v_obs"), axis=0)
    cells = stack_field(rows, ms, "n_cells")
    established = np.nanmean(cells >= ORDER_MIN_MICS, axis=0) >= 0.5
    mean = np.nanmean(net, axis=0)
    n = np.sum(np.isfinite(net), axis=0)
    if len(rows) < 2:
        detected = np.nansum(stack_field(rows, ms, "resolvable").astype(bool), axis=0) > 0
    else:
        sem = np.nanstd(net, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
        detected = (n >= 2) & (mean > 0) & (mean >= DETECT_SIGMA * np.maximum(sem, 1e-12))
    return established & detected & np.isfinite(obs) & (obs <= V_CEIL)


def fit_from_rows(
    rows: list[dict[str, Any]], key: str = "v_net", parity: str = "all"
) -> dict[str, float] | None:
    v, res = {}, {}
    for ms in LAGS_MS:
        v[ms] = np.nanmean(stack_field(rows, ms, key), axis=0)
        res[ms] = consensus(rows, ms)
    return fit_power_law(v, res, parity)


def parity_reads(pooled: dict[str, Any]) -> dict[str, Any]:
    """Odd-to-even ratio of the net per-order residual, per table lag.

    A two-bladed rotor puts most of its sound in the EVEN orders; the odd
    orders are what blade-to-blade dissimilarity leaves over. The two families
    carry very different amounts of independent phase noise, and the ratio is
    the headline of that split.

    Each odd order is compared with the mean of its two EVEN NEIGHBOURS rather
    than with the median of the whole even family, because the detection bar
    removes different orders from each family: comparing family medians would
    compare the surviving odd orders with the surviving even ones, which is a
    comparison between two different `k` ranges.
    """
    out: dict[str, Any] = {}
    for ms in TABLE_LAGS_MS:
        blob = pooled.get(str(ms))
        if not blob:
            continue
        net = np.asarray(blob["v_net"], dtype=np.float64)
        res = np.asarray(blob["resolved"], dtype=bool)
        ok = np.isfinite(net) & (net > 0)
        ratios, pairs = [], []
        for idx in range(FIT_K_LO, FIT_K_HI - 1):
            order = idx + 1
            if order % 2 == 0 or not (res[idx] and ok[idx]):
                continue
            neigh = [
                net[idx - 1] if ok[idx - 1] else np.nan,
                net[idx + 1] if ok[idx + 1] else np.nan,
            ]
            base = float(np.nanmean(neigh))
            if not np.isfinite(base) or base <= 0:
                continue
            ratios.append(net[idx] / base)
            pairs.append((order, float(net[idx]), base))
        out[str(ms)] = {
            "ratio": float(np.median(ratios)) if ratios else float("nan"),
            "n_pairs": len(ratios),
            "v_odd_median": float(np.median([p[1] for p in pairs])) if pairs else float("nan"),
            "v_even_neighbour_median": float(np.median([p[2] for p in pairs]))
            if pairs
            else float("nan"),
        }
    return out


def bootstrap_fit(
    rows: list[dict[str, Any]], key: str = "v_net", parity: str = "all"
) -> dict[str, Any]:
    """Percentile confidence interval of ``(a, p, q)`` over recordings."""
    rng = np.random.default_rng(BOOT_SEED)
    draws: list[dict[str, float]] = []
    n = len(rows)
    for _ in range(N_BOOT):
        pick = [rows[i] for i in rng.integers(0, n, n)]
        f = fit_from_rows(pick, key, parity)
        if f:
            draws.append(f)
    out: dict[str, Any] = {"n_draws": len(draws)}
    for p in ("a", "p", "q"):
        vals = np.asarray([d[p] for d in draws], dtype=np.float64)
        if not vals.size:
            out[p] = {"median": float("nan"), "lo95": float("nan"), "hi95": float("nan")}
            continue
        out[p] = {
            "median": float(np.median(vals)),
            "lo95": float(np.percentile(vals, 2.5)),
            "hi95": float(np.percentile(vals, 97.5)),
        }
    return out


# ═════════════════════════════════════════════════════════════════════════════
# The long-lag question: does V_eps flatten or keep growing?
# ═════════════════════════════════════════════════════════════════════════════
#
# A per-order Wiener phase (the generator the 10-500 ms ladder returned,
# ``V_eps = a k^p tau^q`` with q near 1) has no ceiling: its increment
# variance grows for ever. A per-order OU PHASE has one: ``V`` saturates at
# ``2 sigma_eps^2 / lambda_eps`` after ``tau_c = 1/lambda_eps``. The two are
# indistinguishable below ``tau_c``, so the choice is only decidable at lags
# long enough to reach it - which is what ``--long`` measures.


def fit_growth_power(tau: np.ndarray, v: np.ndarray) -> tuple[dict[str, float], float]:
    """``V = a tau^q`` by least squares on ``log V``; ``(parameters, residual sum)``."""
    design = np.stack([np.ones_like(tau), np.log(tau)], axis=1)
    y = np.log(v)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    rss = float(np.sum((y - design @ coef) ** 2))
    return {"a": float(math.exp(coef[0])), "q": float(coef[1])}, rss


def fit_growth_saturating(tau: np.ndarray, v: np.ndarray) -> tuple[dict[str, float], float]:
    """``V = c (1 - exp(-tau/tau_c))`` by least squares on ``log V``.

    Same response variable and same number of free parameters as the power
    law, so the two residual sums are directly comparable. The fit is
    multi-started over ``tau_c`` because the objective is flat once ``tau_c``
    leaves the measured range.
    """
    from scipy import optimize

    y = np.log(v)

    def resid(p: np.ndarray) -> np.ndarray:
        c, t_c = math.exp(p[0]), math.exp(p[1])
        return np.log(np.maximum(c * -np.expm1(-tau / t_c), 1e-300)) - y

    best: tuple[dict[str, float], float] | None = None
    for t0 in (0.1, 0.3, 1.0, 3.0, 10.0, 100.0):
        p0 = np.array([math.log(max(float(v.max()), 1e-12)), math.log(t0)])
        sol = optimize.least_squares(resid, p0, bounds=([-40.0, -9.0], [40.0, 14.0]))
        rss = float(np.sum(np.asarray(sol.fun) ** 2))
        if best is None or rss < best[1]:
            best = ({"c": float(math.exp(sol.x[0])), "tau_c": float(math.exp(sol.x[1]))}, rss)
    assert best is not None
    return best


def _aic(rss: float, n: int) -> float:
    """Gaussian AIC of a two-parameter fit with an estimated variance."""
    return n * math.log(max(rss, 1e-300) / n) + 2.0 * 3.0


def compare_growth(
    tau: np.ndarray, v: np.ndarray, obs: np.ndarray, *, ceil_trust: float = math.inf
) -> dict[str, Any]:
    """Both long-lag shapes on one order's curve, with the AIC difference and a verdict.

    ``obs`` is the OBSERVED residual of the same cells. Above :data:`V_CEIL`
    the circular estimator has no headroom left - ``|gamma|`` has reached its
    own sampling floor - so such a lag reports a bend that belongs to the
    instrument, not to the process, and is excluded from the fit.

    ``ceil_trust`` is the plateau above which a saturating fit means nothing,
    measured on control (ii): that control carries a planted random walk and
    therefore has NO ceiling, so the smallest plateau this comparison invents
    for it is the level at which the estimator's own compression already looks
    like saturation. A fitted plateau at or above it is reported as
    undetermined, not as a ceiling.
    """
    ok = (
        np.isfinite(v)
        & (v > 0.0)
        & (tau >= LONG_FIT_TAU_LO_MS * 1e-3 - 1e-9)
        & (np.isfinite(obs) & (obs <= V_CEIL))
    )
    n = int(ok.sum())
    n_ceil = int((np.isfinite(obs) & (obs > V_CEIL) & (tau >= LONG_FIT_TAU_LO_MS * 1e-3)).sum())
    if n < LONG_FIT_MIN_POINTS:
        return {"n_points": n, "n_lags_at_ceiling": n_ceil, "verdict": "no fit"}
    t, y = tau[ok], v[ok]
    power, rss_p = fit_growth_power(t, y)
    sat, rss_s = fit_growth_saturating(t, y)
    aic_p, aic_s = _aic(rss_p, n), _aic(rss_s, n)
    d_aic = aic_p - aic_s
    # A time constant beyond the longest lag fitted means the saturating
    # curve has straightened into the power law inside the data: it may fit
    # marginally better, but it is not evidence of a ceiling.
    reached = sat["tau_c"] <= float(t.max())
    growth = float(y[np.argmax(t)] / y[np.argmin(t)])
    if growth <= FLAT_GROWTH_RATIO:
        # No tau dependence left to choose a shape on: the curve is already at
        # its plateau over the whole fitted range (or is a flat floor).
        verdict = "flat"
    elif d_aic > AIC_DECISIVE and reached:
        verdict = "ceiling"
    elif d_aic < -AIC_DECISIVE:
        verdict = "growing"
    elif not reached:
        verdict = "growing (no ceiling inside the fitted lags)"
    else:
        verdict = "undetermined"
    trusted = sat["c"] < ceil_trust
    if verdict in ("ceiling", "flat") and not trusted:
        verdict = "undetermined (plateau where control (ii) fakes one)"
    return {
        "n_points": n,
        "n_lags_at_ceiling": n_ceil,
        "tau_lo_s": float(t.min()),
        "tau_hi_s": float(t.max()),
        "growth_ratio": growth,
        "power": {**power, "rss": rss_p, "aic": aic_p},
        "saturating": {**sat, "rss": rss_s, "aic": aic_s},
        "d_aic": float(d_aic),
        "tau_c_reached": bool(reached),
        "ceil_trust": ceil_trust,
        "plateau_trusted": bool(trusted),
        "verdict": verdict,
    }


def _lag_value(row: dict[str, Any], ms: int, key: str, k: int) -> float:
    entry = row["orders"].get(str(ms))
    if not entry or key not in entry:
        return float("nan")
    return float(entry[key][k - 1])


def long_orders(rate: float) -> list[int]:
    """The orders of :data:`K_FAMILY` whose line sits below the anti-alias corner."""
    return [k for k in K_FAMILY if k * rate <= AA_CORNER_HZ]


def long_curves(
    rows: list[dict[str, Any]],
    key: str,
    lags_ms: tuple[int, ...],
    *,
    ceil_trust: float = math.inf,
) -> dict[str, Any]:
    """Per-order long-lag curves and shape comparisons, per support and pooled.

    ``key`` selects the quantity read: ``v_net`` for the data and for the
    planted control (floor-subtracted), ``v_leak_corrected`` for the floor
    control, whose residual IS the floor and whose net is identically zero.
    """
    tau = np.asarray([ms * 1e-3 for ms in lags_ms], dtype=np.float64)
    per_support: list[dict[str, Any]] = []
    for row in rows:
        orders: dict[str, Any] = {}
        for k in long_orders(float(row["rate_rps"])):
            v = np.asarray([_lag_value(row, ms, key, k) for ms in lags_ms], dtype=np.float64)
            obs_curve = np.asarray(
                [_lag_value(row, ms, "v_obs", k) for ms in lags_ms], dtype=np.float64
            )
            obs = _lag_value(row, LONG_CEIL_LAG_MS, "v_obs", k)
            orders[str(k)] = {
                "v": v.tolist(),
                "v_obs": obs_curve.tolist(),
                "v_table": {str(ms): float(v[i]) for i, ms in enumerate(lags_ms)},
                "err_table": {
                    str(ms): _lag_value(row, ms, "net_err", k)
                    for ms in lags_ms
                    if ms in LONG_TABLE_LAGS_MS
                },
                "ceil_table": {
                    str(ms): bool(np.isfinite(obs_curve[i]) and obs_curve[i] > V_CEIL)
                    for i, ms in enumerate(lags_ms)
                },
                "v_obs_ceil_lag": obs,
                "at_ceiling": bool(np.isfinite(obs) and obs > V_CEIL),
                "fit": compare_growth(tau, v, obs_curve, ceil_trust=ceil_trust),
            }
        per_support.append(
            {
                "id": row["id"],
                "rate_rps": row["rate_rps"],
                "seg_len_s": row["seg_len_s"],
                "window_source": row.get("window_source"),
                "k_max_below_corner": long_orders(float(row["rate_rps"]))[-1],
                "orders": orders,
            }
        )
    pooled: dict[str, Any] = {}
    for k in sorted({int(kk) for s in per_support for kk in s["orders"]}):
        stack = np.stack(
            [
                np.asarray(s["orders"][str(k)]["v"], dtype=np.float64)
                for s in per_support
                if str(k) in s["orders"]
            ]
        )
        obs_stack = np.stack(
            [
                np.asarray(s["orders"][str(k)]["v_obs"], dtype=np.float64)
                for s in per_support
                if str(k) in s["orders"]
            ]
        )
        with np.errstate(invalid="ignore"):
            v = np.nanmean(stack, axis=0)
            obs_curve = np.nanmean(obs_stack, axis=0)
        obs_mean = float(obs_curve[lags_ms.index(LONG_CEIL_LAG_MS)])
        pooled[str(k)] = {
            "v": v.tolist(),
            "v_obs": obs_curve.tolist(),
            "v_table": {str(ms): float(v[i]) for i, ms in enumerate(lags_ms)},
            "ceil_table": {
                str(ms): bool(np.isfinite(obs_curve[i]) and obs_curve[i] > V_CEIL)
                for i, ms in enumerate(lags_ms)
            },
            "n_supports": int(stack.shape[0]),
            "v_obs_ceil_lag": obs_mean,
            "at_ceiling": bool(np.isfinite(obs_mean) and obs_mean > V_CEIL),
            "fit": compare_growth(tau, v, obs_curve, ceil_trust=ceil_trust),
        }
    return {"per_support": per_support, "pooled": pooled}


def long_verdict(sources: dict[str, Any]) -> str:
    """The honest pooled answer, read off the pooled fits and calibrated on the controls."""
    data = sources["data"]["pooled"]
    free = sorted(
        (k for k, blob in data.items() if not blob["at_ceiling"] and "power" in blob["fit"]),
        key=int,
    )
    counts: dict[str, list[str]] = {}
    for k in free:
        counts.setdefault(data[k]["fit"]["verdict"].split(" (")[0], []).append(k)
    floor = sources["matched"]["pooled"]
    plant = sources["matched_planted"]["pooled"]

    def read(blob: dict[str, Any]) -> str:
        rows = [
            (k, b["fit"])
            for k, b in sorted(blob.items(), key=lambda kv: int(kv[0]))
            if b["fit"].get("n_points", 0) >= LONG_FIT_MIN_POINTS and not b["at_ceiling"]
        ]
        if not rows:
            return "no fittable order"
        q = float(np.median([f["power"]["q"] for _, f in rows]))
        ver = sorted({f["verdict"].split(" (")[0] for _, f in rows})
        return f"median q = {q:+.2f}, verdicts {'/'.join(ver)} over {len(rows)} orders"

    if not free:
        return (
            "Every order of the long support set is already at the estimator ceiling by "
            f"{LONG_CEIL_LAG_MS} ms, so the long lags carry no information about the shape and "
            "the question is UNDETERMINED on this bench."
        )
    parts = [f"orders {', '.join(k for k in free)} are below the ceiling at {LONG_CEIL_LAG_MS} ms"]
    for name, ks in sorted(counts.items(), key=lambda kv: -len(kv[1])):
        parts.append(f"{name}: k = {', '.join(sorted(ks, key=int))}")
    return (
        "; ".join(parts)
        + f". Calibration - control (i) (shaft + floor only, must be flat): {read(floor)}; "
        f"control (ii) (planted D_k = {CTRL_D1} k random walk, must keep growing): {read(plant)}."
    )


def _order_sentence(k: int, blob: dict[str, Any]) -> str:
    """One order's answer, with the parameter the winning shape implies."""
    fit = blob["fit"]
    tab = blob["v_table"]

    def at(ms: int) -> str:
        val = tab.get(str(ms))
        return _f(val) if val is not None else "-"

    head = (
        f"**k = {k}**: V_eps = {at(500)} rad^2 at 0.5 s, {at(1000)} at 1 s, {at(2000)} at 2 s, "
        f"{at(5000)} at 5 s"
    )
    if blob["at_ceiling"]:
        return (
            f"{head} - but its observed residual is already at the {V_CEIL:.0f} rad^2 estimator "
            f"ceiling at {LONG_CEIL_LAG_MS} ms, so the shape cannot be read and the answer is "
            "UNDETERMINED for this order."
        )
    if "power" not in fit:
        return (
            f"{head} - only {fit['n_points']} of the fitted lags are readable "
            f"({fit['n_lags_at_ceiling']} of them sit over the {V_CEIL:.0f} rad^2 estimator "
            f"ceiling), fewer than the {LONG_FIT_MIN_POINTS} a two-parameter shape needs, so no "
            "fit and the answer is UNDETERMINED for this order."
        )
    q = fit["power"]["q"]
    t_c, c, d = fit["saturating"]["tau_c"], fit["saturating"]["c"], fit["d_aic"]
    core = f"{head}; q = {q:+.2f}, tau_c = {t_c:.2f} s, dAIC = {d:+.1f}"
    verdict = fit["verdict"]
    if verdict == "ceiling":
        lam = 1.0 / t_c
        return (
            f"{core} - the saturating shape wins by more than {AIC_DECISIVE:.0f} AIC and its knee "
            f"is inside the measured lags, so this order has a CEILING: an OU phase with "
            f"lambda_eps = {lam:.3g} /s and sigma_eps = {math.sqrt(max(c * lam, 0.0)):.3g} rad/s^0.5 "
            f"(plateau {c:.3g} rad^2)."
        )
    if verdict.startswith("growing"):
        why = (
            f"the power law wins by {-d:.1f} AIC"
            if d < -AIC_DECISIVE
            else f"the saturating fit only reaches its knee at tau_c = {t_c:.2g} s, beyond the "
            "longest lag measured, i.e. it has straightened into the power law inside the data"
        )
        return (
            f"{core} - {why}, so this order KEEPS GROWING out to the longest lag; as a Wiener "
            f"phase that is D_k = {0.5 * fit['power']['a']:.3g} rad^2/s at q = {q:.2f}."
        )
    if verdict == "flat":
        return (
            f"{core} - the curve grows by only a factor {fit['growth_ratio']:.2f} across the "
            f"fitted lags, so it carries no `tau` dependence at all: this order is already AT its "
            f"plateau by {fit['tau_lo_s']:.1f} s, which is a ceiling reached before the long lags "
            f"begin (plateau about {c:.3g} rad^2)."
        )
    if verdict.startswith("undetermined (plateau"):
        return (
            f"{core} - a saturating shape does fit it (plateau {c:.3g} rad^2), but control (ii), "
            f"which has NO ceiling by construction, is fitted one at {fit['ceil_trust']:.3g} rad^2 "
            "through the same estimator, so a plateau at this level is the read-out compressing "
            "and not the process saturating: UNDETERMINED for this order."
        )
    return (
        f"{core} - the two shapes fit equally well (|dAIC| <= {AIC_DECISIVE:.0f}) and the "
        f"saturating knee sits at tau_c = {t_c:.2g} s, so this order is UNDETERMINED: these lags "
        "on these windows do not separate a ceiling from continued growth."
    )


def long_paragraph(sources: dict[str, Any]) -> str:
    """The per-order answers and the pooled one, in prose."""
    data = sources["data"]["pooled"]
    ks = sorted((int(k) for k in data), key=int)
    lines = [_order_sentence(k, data[str(k)]) for k in ks]
    free = [k for k in ks if not data[str(k)]["at_ceiling"] and "power" in data[str(k)]["fit"]]
    tally: dict[str, list[int]] = {}
    for k in free:
        tally.setdefault(data[str(k)]["fit"]["verdict"].split(" (")[0], []).append(k)
    if not free:
        pooled = (
            f"Pooled: no order of this support set is below the estimator ceiling at "
            f"{LONG_CEIL_LAG_MS} ms, so the long lags cannot answer the question at all."
        )
    else:
        got = "; ".join(
            f"{name} at k = {', '.join(str(k) for k in ks_)}"
            for name, ks_ in sorted(tally.items(), key=lambda kv: -len(kv[1]))
        )
        dominant = max(tally.items(), key=lambda kv: len(kv[1]))[0]
        shape = {
            "ceiling": "the independent per-order phase is an OU phase with a finite plateau, "
            "not a random walk",
            "flat": "the independent per-order term has no lag dependence left at these lags - "
            "it is at its plateau before 1 s, which is a ceiling reached earlier than this run "
            "can resolve",
            "growing": "the independent per-order phase keeps diffusing out to the longest lag "
            "fitted, i.e. the Wiener-per-order generator is not contradicted",
            "undetermined": "these lags do not separate the two, so the choice must be made on "
            "grounds other than this measurement",
        }[dominant]
        pooled = (
            f"Pooled over the {len(free)} orders that are below the ceiling at "
            f"{LONG_CEIL_LAG_MS} ms ({got}), the answer is **{dominant.upper()}**: {shape}."
        )
    floor = sources["matched"]["pooled"]
    plant = sources["matched_planted"]["pooled"]

    def median_q(blob: dict[str, Any]) -> float:
        qs = [
            b["fit"]["power"]["q"]
            for b in blob.values()
            if "power" in b["fit"] and not b["at_ceiling"]
        ]
        return float(np.median(qs)) if qs else float("nan")

    trust = ceiling_trust_level(sources["matched_planted"])
    calib = (
        f"The instrument is calibrated by the two controls at the same lags: control (i), which "
        f"carries no per-order term at all, returns a floor with median q = "
        f"{median_q(floor):+.2f} (flat is q = 0), and control (ii), which carries a planted "
        f"random walk, returns median q = {median_q(plant):+.2f} (a random walk is q = 1). "
        + (
            f"Control (ii) also sets the level above which a saturating fit means nothing: it has "
            f"no ceiling by construction, yet this comparison fits it one at a plateau of "
            f"{trust:.3g} rad^2, so any plateau at or above that is the estimator's own "
            "compression and is reported as undetermined."
            if math.isfinite(trust)
            else "Control (ii) is never fitted a ceiling at these levels, so no plateau had to be "
            "discounted as the estimator's own compression."
        )
    )
    return "\n\n".join([*lines, pooled, calib])


def ceiling_trust_level(planted: dict[str, Any]) -> float:
    """The lowest plateau at which the estimator FAKES a ceiling, from control (ii).

    Control (ii) carries a planted per-order random walk and therefore has no
    ceiling at all. Any of its orders that the shape comparison nevertheless
    calls a ceiling is the circular estimator compressing as ``V`` approaches
    its own read-out limit; the smallest such plateau is the level above which
    a saturating fit on the data carries no information. ``inf`` when the
    control is never fitted a ceiling, i.e. nothing has to be discounted.
    """
    cs = [
        b["fit"]["saturating"]["c"]
        for b in planted["pooled"].values()
        if b["fit"].get("verdict") == "ceiling"
    ]
    return float(min(cs)) if cs else math.inf


def long_lag_reads(
    groups: dict[str, list[dict[str, Any]]], lags_ms: tuple[int, ...]
) -> dict[str, Any]:
    """The whole long-lag block: data, both controls, the fits and the verdict."""
    # The controls are read FIRST and with no trust bar of their own, because
    # control (ii) is what measures the bar the data is then read against.
    floor = long_curves(groups["matched"], "v_leak_corrected", lags_ms)
    plant = long_curves(groups["matched_planted"], "v_net", lags_ms)
    trust = ceiling_trust_level(plant)
    sources = {
        "data": long_curves(groups["data"], "v_net", lags_ms, ceil_trust=trust),
        "matched": floor,
        "matched_planted": plant,
    }
    return {
        "lags_ms": list(lags_ms),
        "tau_s": [ms * 1e-3 for ms in lags_ms],
        "table_lags_ms": list(LONG_TABLE_LAGS_MS),
        "fit_tau_lo_ms": LONG_FIT_TAU_LO_MS,
        "ceil_lag_ms": LONG_CEIL_LAG_MS,
        "aic_decisive": AIC_DECISIVE,
        "flat_growth_ratio": FLAT_GROWTH_RATIO,
        "ceiling_trust_plateau": trust if math.isfinite(trust) else None,
        "sources": sources,
        "verdict": long_verdict(sources),
        "verdict_paragraph": long_paragraph(sources),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════════════════════


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure {_rel(path)}")


K_FAMILY = (2, 4, 8, 16, 24, 32, 40)


def fig_structure(res: dict[str, Any], path: Path) -> None:
    pooled = res["pooled"]
    lags = sorted(int(k) for k in pooled)
    tau = np.asarray([pooled[str(m)]["tau_s"] for m in lags])
    cmap = plt.get_cmap("viridis")
    panels = (
        ("DREGON bench: $V_\\varepsilon$ net", "pooled", "v_net"),
        (
            "control (i): shaft + floor only\n(the measured floor, must be flat)",
            "pooled_control_matched",
            "v_leak_corrected",
        ),
        (
            "control (ii): $+\\,D_k = 0.1\\,k$\n(net; dashed = planted $2D_k\\tau$)",
            "pooled_control_matched_planted",
            "v_net",
        ),
    )
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=True)
    for ax, (title, src, key) in zip(axs, panels):
        blob = res.get(src)
        if not blob:
            ax.set_visible(False)
            continue
        for i, k in enumerate(K_FAMILY):
            colour = cmap(i / max(len(K_FAMILY) - 1, 1))
            y = np.asarray([blob[str(m)][key][k - 1] for m in lags], dtype=np.float64)
            ok = np.isfinite(y) & (y > 0)
            ax.plot(tau[ok], y[ok], "o-", ms=3.5, lw=1.3, color=colour, label=f"k={k}")
            if src.endswith("planted"):
                ax.plot(tau, 2.0 * CTRL_D1 * k * tau, "--", lw=1.0, color=colour, alpha=0.8)
        vsh = np.asarray([blob[str(m)]["v_shaft"] for m in lags])
        ax.plot(tau, vsh, "k-", lw=2.4, label=r"$V_{\rm shared}$")
        ax.plot(tau, 1600.0 * vsh, "k:", lw=1.4, label=r"$40^2 V_{\rm shared}$")
        ax.plot(
            tau,
            ou_structure(tau, SHAFT_SIGMA_NU, SHAFT_LAM),
            color="crimson",
            ls=":",
            lw=1.8,
            label=f"OU({SHAFT_SIGMA_NU}, {SHAFT_LAM})",
        )
        ax.axhline(V_CEIL, color="grey", lw=0.9, ls="-.")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"lag $\tau$ [s]")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel(r"variance [rad$^2$]")
    axs[0].set_ylim(1e-3, 3e1)
    axs[0].legend(fontsize=7, ncol=2, loc="upper left")
    fig.suptitle(
        "Independent per-order phase noise $V_\\varepsilon(k,\\tau)$ against the shared shaft term "
        "(grey dash-dot: estimator ceiling)",
        fontsize=11,
    )
    _save(fig, path)


def fig_vs_k(res: dict[str, Any], path: Path) -> None:
    pooled = res["pooled"]
    fit = res["fit"]["net"]
    k = np.arange(1, K_MAX + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, ms in zip(axs, TABLE_LAGS_MS):
        blob = pooled.get(str(ms))
        if not blob:
            ax.set_visible(False)
            continue
        tau = blob["tau_s"]
        net = np.asarray(blob["v_net"], dtype=np.float64)
        obs = np.asarray(blob["v_obs"], dtype=np.float64)
        floor = np.asarray(blob["v_floor"], dtype=np.float64)
        shared = blob["v_shaft"] * k**2
        ax.plot(k, obs, "s-", ms=3, lw=1.0, color="0.55", label=r"$V_\varepsilon$ observed")
        ax.plot(k, floor, "^-", ms=3, lw=1.0, color="tab:orange", label="noise floor (control i)")
        ax.plot(k, net, "o-", ms=4.5, lw=1.8, color="tab:blue", label=r"$V_\varepsilon$ net")
        ax.plot(k, shared, "k--", lw=1.6, label=r"$k^2 V_{\rm shared}$")
        if fit:
            ax.plot(
                k,
                fit["a"] * k ** fit["p"] * tau ** fit["q"],
                color="crimson",
                lw=1.6,
                ls=":",
                label=f"fit $a k^{{{fit['p']:.2f}}}\\tau^{{{fit['q']:.2f}}}$",
            )
        ax.axhline(V_CEIL, color="grey", lw=0.8, ls="-.")
        ax.axvspan(0.5, FIT_K_LO - 0.5, color="0.9", zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("order $k$")
        ax.set_title(f"$\\tau$ = {ms} ms", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel(r"variance [rad$^2$]")
    axs[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Per-order phase-noise variance against order (grey band: below the fit range)", fontsize=11
    )
    _save(fig, path)


def fig_corr(res: dict[str, Any], path: Path) -> None:
    corr = np.asarray(res["correlations"]["order_corr"], dtype=np.float64)
    usable = np.asarray(res["correlations"]["order_usable"], dtype=bool)
    mic = np.asarray(res["correlations"]["mic_rho"], dtype=np.float64)
    mic_ctrl = res["correlations"].get("mic_rho_controls", {})
    fig = plt.figure(figsize=(13.5, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.15, 1.0, 1.25), wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    show = corr.copy()
    show[~np.outer(usable, usable)] = np.nan
    im = ax.imshow(
        show,
        origin="lower",
        extent=(0.5, K_MAX + 0.5, 0.5, K_MAX + 0.5),
        vmin=-0.3,
        vmax=0.3,
        cmap="RdBu_r",
    )
    fig.colorbar(im, ax=ax, label=r"$\rho(\varepsilon_k, \varepsilon_l)$")
    ax.set_xlabel("order $l$")
    ax.set_ylabel("order $k$")
    ax.set_title(f"cross-order, $\\tau$ = {CORR_LAG_MS} ms", fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    blocks = res["correlations"]["blocks"]
    names = ["off-diag", "same parity", "cross parity", r"$|k-l|=2$"]
    vals = [
        blocks["off_diagonal"],
        blocks["same_parity"],
        blocks["cross_parity"],
        blocks["neighbour_dk2"],
    ]
    ax.bar(names, vals, color=["tab:blue", "tab:green", "tab:red", "tab:purple"])
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(CORR_INDEP_MAX, color="grey", ls=":", lw=1.0)
    ax.axhline(-CORR_INDEP_MAX, color="grey", ls=":", lw=1.0)
    ax.set_ylabel(r"mean $\rho$")
    ax.set_title("cross-order block means", fontsize=10)
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[0, 2])
    k = np.arange(1, K_MAX + 1)
    ax.plot(k, mic, "o-", ms=4, lw=1.6, color="tab:blue", label="DREGON bench")
    for name, style, lab in (
        ("matched", "^--", "control (i): noise only"),
        ("matched_planted", "s:", "control (ii): shared $D_k$"),
    ):
        y = mic_ctrl.get(name)
        if y:
            ax.plot(k, np.asarray(y, dtype=np.float64), style, ms=3, lw=1.2, label=lab)
    ax.axhline(CORR_SHARED_MIN, color="grey", ls=":", lw=1.0)
    ax.axhline(CORR_INDEP_MAX, color="grey", ls=":", lw=1.0)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("order $k$")
    ax.set_ylabel(r"mean cross-microphone $\rho$")
    ax.set_ylim(-0.25, 1.05)
    ax.set_title(f"cross-microphone, $\\tau$ = {CORR_LAG_MS} ms", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(
        "Is the per-order residual independent across orders and across microphones?", fontsize=11
    )
    _save(fig, path)


def fig_long(res: dict[str, Any], path: Path) -> None:
    """The long-lag curves of the data and of both controls, with the two fits."""
    long = res["long_lag"]
    tau = np.asarray(long["tau_s"], dtype=np.float64)
    fine = np.geomspace(max(tau.min(), 1e-3), tau.max(), 200)
    cmap = plt.get_cmap("viridis")
    panels = (
        ("DREGON single-rotor bench: $V_\\varepsilon$ net", "data"),
        ("control (i): shaft + floor only\n(the measured floor, must be flat)", "matched"),
        (
            f"control (ii): $+\\,D_k = {CTRL_D1}\\,k$ random walk\n"
            "(net; dashed = planted $2D_k\\tau$)",
            "matched_planted",
        ),
    )
    fig, axs = plt.subplots(1, 3, figsize=(16.0, 5.4), sharey=True)
    for ax, (title, src) in zip(axs, panels):
        pooled = long["sources"][src]["pooled"]
        ks = sorted((int(k) for k in pooled), key=int)
        for i, k in enumerate(ks):
            blob = pooled[str(k)]
            colour = cmap(i / max(len(ks) - 1, 1))
            y = np.asarray(blob["v"], dtype=np.float64)
            over = np.asarray(blob["v_obs"], dtype=np.float64) > V_CEIL
            ok = np.isfinite(y) & (y > 0)
            # Filled markers are cells the estimator can still read; open ones
            # sit over its ceiling and take part in no fit.
            ax.plot(tau[ok], y[ok], "-", lw=1.3, color=colour, label=f"k={k}")
            ax.plot(tau[ok & ~over], y[ok & ~over], "o", ms=3.5, color=colour)
            ax.plot(
                tau[ok & over],
                y[ok & over],
                "o",
                ms=4.5,
                mfc="none",
                mec=colour,
                mew=1.0,
            )
            if src == "matched_planted":
                ax.plot(fine, 2.0 * CTRL_D1 * k * fine, "--", lw=1.0, color=colour, alpha=0.8)
            fit = blob["fit"]
            if "power" not in fit:
                continue
            # The fitted shapes are drawn only where they were fitted.
            band = np.geomspace(fit["tau_lo_s"], fit["tau_hi_s"], 100)
            ax.plot(
                band,
                fit["power"]["a"] * band ** fit["power"]["q"],
                lw=1.2,
                ls=(0, (4, 2)),
                color=colour,
                alpha=0.8,
            )
            ax.plot(
                band,
                fit["saturating"]["c"] * -np.expm1(-band / fit["saturating"]["tau_c"]),
                lw=1.2,
                ls=(0, (1, 1.5)),
                color=colour,
                alpha=0.8,
            )
        ax.axhline(V_CEIL, color="grey", lw=0.9, ls="-.")
        ax.axvspan(tau.min(), LONG_FIT_TAU_LO_MS * 1e-3, color="0.95", zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"lag $\tau$ [s]")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel(r"variance [rad$^2$]")
    axs[0].set_ylim(1e-3, 3e1)
    axs[0].legend(fontsize=7, ncol=2, loc="upper left")
    fig.suptitle(
        "Long-lag shape of the independent per-order term: dashed = fitted "
        "$a\\tau^q$, dotted = fitted $c(1-e^{-\\tau/\\tau_c})$ (open markers: observed residual "
        "over the estimator ceiling, excluded from both fits; grey dash-dot: that ceiling; "
        "grey band: below the fit range)",
        fontsize=11,
    )
    _save(fig, path)


# ═════════════════════════════════════════════════════════════════════════════
# Report
# ═════════════════════════════════════════════════════════════════════════════


def _f(x: Any, fmt: str = "{:.3g}") -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return fmt.format(x) if isinstance(x, (int, float)) else str(x)


def summary_table(res: dict[str, Any]) -> list[list[str]]:
    pooled = res["pooled"]
    a = pooled[str(TABLE_LAGS_MS[0])]
    b = pooled[str(TABLE_LAGS_MS[1])]
    head = [
        "k",
        f"V_eps({TABLE_LAGS_MS[0]} ms)",
        f"V_eps({TABLE_LAGS_MS[1]} ms)",
        f"Var[k dtheta] ({TABLE_LAGS_MS[0]} ms)",
        "ratio",
    ]
    rows = [head]
    for k in range(1, K_MAX + 1):
        va = a["v_net"][k - 1]
        vb = b["v_net"][k - 1]
        shared = a["v_shaft"] * k * k
        ratio = va / shared if shared > 0 and math.isfinite(va) else float("nan")
        rows.append(
            [str(k), _f(va, "{:.4g}"), _f(vb, "{:.4g}"), _f(shared, "{:.4g}"), _f(ratio, "{:.3g}")]
        )
    return rows


def markdown_table(rows: list[list[str]]) -> str:
    head, body = rows[0], rows[1:]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def control_table(res: dict[str, Any]) -> list[list[str]]:
    rec = res["control_recovery"]["matched"]
    rows = [
        [
            "k",
            "tau [ms]",
            "planted 2 D_k tau",
            "recovered",
            "ratio",
            "floor (ctrl i)",
            "floor analytic",
        ]
    ]
    for r in rec["cells"]:
        rows.append(
            [
                str(r["k"]),
                str(r["lag_ms"]),
                _f(r["planted"], "{:.4g}"),
                _f(r["recovered"], "{:.4g}"),
                _f(r["ratio"], "{:.3g}"),
                _f(r["floor"], "{:.4g}"),
                _f(r["floor_analytic"], "{:.4g}"),
            ]
        )
    return rows


def write_findings(res: dict[str, Any], out_dir: Path) -> Path:
    pooled = res["pooled"]
    fit = res["fit"]["net"]
    boot = res["fit"]["net_bootstrap"]
    fit_obs = res["fit"]["obs"]
    blocks = res["correlations"]["blocks"]
    rec = res["control_recovery"]["matched"]
    rec_nom = res["control_recovery"]["nominal"]
    lag_a, lag_b = TABLE_LAGS_MS
    a = pooled[str(lag_a)]
    b = pooled[str(lag_b)]
    mic = np.asarray(res["correlations"]["mic_rho"], dtype=np.float64)
    mic_mid = float(np.nanmean(mic[MID_K]))

    lines: list[str] = []
    lines.append("# Order decoherence on the DREGON bench - findings\n")
    lines.append(
        "Source: `scripts/noise_v2_order_decoherence.py`. Numbers: "
        "`results/noise_v2/decoherence/decoherence.json`. Figures: "
        "`docs/explainers/noise-model-v2-plan/decoherence_*.png`.\n"
    )
    lines.append(
        "The model under test splits the phase increment of order `k` over a lag `tau` into a "
        "SHARED part and an INDEPENDENT part:\n\n"
        "    dphi_k(t; tau) = k * dtheta(t; tau) + eps_k(t; tau)\n\n"
        "`dtheta` is everything that scales with the order - the shaft speed error and any "
        "propagation term tied to it. `eps_k` is what the tied model cannot represent. "
        f"`V_shared(tau) = Var[dtheta]` and `V_eps(k, tau) = Var[eps_k]` are measured on "
        f"{res['n_recordings']} single-motor bench recordings, 8 microphones, orders 1-{K_MAX}, on "
        f"a {FS_PHASE:.1f} Hz phase grid.\n"
    )

    lines.append("## The numbers\n")
    lines.append(markdown_table(summary_table(res)) + "\n")
    lines.append(
        f"`V_eps` is the net value: observed, minus the measured leakage of the shaft estimate, "
        f"minus the additive-noise floor of the paired control. `Var[k dtheta]` is "
        f"`k^2 V_shared({lag_a} ms)` with `V_shared({lag_a} ms) = {a['v_shaft']:.3e}` rad^2 "
        f"(`V_shared({lag_b} ms) = {b['v_shaft']:.3e}` rad^2). `ratio` is "
        f"`V_eps({lag_a} ms) / Var[k dtheta]({lag_a} ms)` - how much of the per-order phase noise "
        "the tied model misses. A dash is an order with no gated cell at that lag; a non-positive "
        "entry is an order whose residual sits AT the additive-noise floor, so no independent "
        "term is resolved there.\n"
    )

    lines.append("## The power law\n")
    lines.append(
        f"`V_eps(k, tau) = a k^p tau^q` by log-log least squares over k in "
        f"[{FIT_K_LO}, {FIT_K_HI}] and tau in [{FIT_TAU_LO_MS:.0f}, {FIT_TAU_HI_MS:.0f}] ms, "
        f"{fit['n_cells']} resolvable cells, rms log residual {fit['rms_log_resid']:.3f}. A cell "
        f"is resolvable when its order clears the gate on at least {ORDER_MIN_MICS} of the 8 "
        f"microphones in most recordings, its observed residual is below the {V_CEIL} rad^2 "
        f"estimator ceiling, and its recording-mean net value stands {DETECT_SIGMA:.0f} standard "
        "errors ACROSS recordings clear of zero:\n"
    )
    lines.append(
        f"- **a = {fit['a']:.4g}** (95 % CI {boot['a']['lo95']:.4g} - {boot['a']['hi95']:.4g})\n"
        f"- **p = {fit['p']:.3f}** (95 % CI {boot['p']['lo95']:.3f} - {boot['p']['hi95']:.3f})\n"
        f"- **q = {fit['q']:.3f}** (95 % CI {boot['q']['lo95']:.3f} - {boot['q']['hi95']:.3f})\n"
    )
    lines.append(
        f"The CI is a {boot['n_draws']}-draw bootstrap over the {res['n_recordings']} recordings. "
        f"Without the floor subtraction the same fit gives a = {fit_obs['a']:.4g}, "
        f"p = {fit_obs['p']:.3f}, q = {fit_obs['q']:.3f}; the floor is flat in `tau`, so leaving "
        "it in flattens `q` and steepens `p`.\n"
    )
    par = res["parity"]
    ratio_a = par[str(lag_a)]["ratio"]
    lines.append(
        f"The rotor has two blades, so its sound sits in the EVEN orders and the odd orders are "
        f"what blade-to-blade dissimilarity leaves over - and the two families do not carry the "
        f"same independent phase noise. Each resolvable odd order carries {ratio_a:.1f} times the "
        f"independent phase noise of the mean of its two even neighbours at {lag_a} ms "
        f"({par[str(lag_a)]['n_pairs']} such orders, median "
        f"{par[str(lag_a)]['v_odd_median']:.3g} against "
        f"{par[str(lag_a)]['v_even_neighbour_median']:.3g} rad^2), and "
        f"{par[str(lag_b)]['ratio']:.1f} times at {lag_b} ms. That split, not scatter, is a good "
        f"part of the {fit['rms_log_resid']:.2f} rms log residual of the joint fit. Fitted "
        "separately:\n"
    )
    for name, parity in (("even", "even"), ("odd", "odd")):
        f_p = res["fit"][f"net_{parity}"]
        b_p = res["fit"][f"net_{parity}_bootstrap"]
        if not f_p:
            lines.append(
                f"- **{name} orders**: fewer than 6 resolvable cells in the fit domain, no fit.\n"
            )
            continue
        lines.append(
            f"- **{name} orders** ({f_p['n_cells']} cells, rms {f_p['rms_log_resid']:.3f}): "
            f"a = {f_p['a']:.4g} ({b_p['a']['lo95']:.4g} - {b_p['a']['hi95']:.4g}), "
            f"p = {f_p['p']:.3f} ({b_p['p']['lo95']:.3f} - {b_p['p']['hi95']:.3f}), "
            f"q = {f_p['q']:.3f} ({b_p['q']['lo95']:.3f} - {b_p['q']['hi95']:.3f})\n"
        )

    lines.append("## Is the residual independent?\n")
    lines.append(
        f"Cross-order correlation of `eps_k` at {CORR_LAG_MS} ms, over "
        f"{blocks['n_pairs']} order pairs (rows de-rotated with one shaft estimate, columns with a "
        "disjoint one, so the shaft-estimate error cannot manufacture a correlation):\n"
    )
    lines.append(
        f"- mean off-diagonal `rho` = **{blocks['off_diagonal']:.3f}** "
        f"(mean |rho| = {blocks['off_diagonal_abs']:.3f})\n"
        f"- same parity (odd-odd and even-even) = {blocks['same_parity']:.3f}, "
        f"cross parity = {blocks['cross_parity']:.3f}, `|k-l| = 2` = {blocks['neighbour_dk2']:.3f}\n"
    )
    lines.append(f"**Verdict:** {res['correlations']['order_verdict']}\n")
    lines.append(
        f"Cross-microphone correlation of `eps_k` at the same order and lag, averaged over the 28 "
        f"microphone pairs: mean over k = 4-24 is **{mic_mid:.3f}**.\n"
    )
    lines.append(f"**Verdict:** {res['correlations']['mic_verdict']}\n")

    lines.append("## Controls\n")
    sig = [r["control_provenance"]["sigma_matched"] for r in res["recordings"]]
    lines.append(
        "Every control is rendered at the rate, the order count and the per-(order, microphone) "
        "line-to-noise ratio of its own recording and is pushed through the identical pipeline. "
        f"Control (i) carries ONLY a shared integrated-OU shaft "
        f"(sigma_nu = {SHAFT_SIGMA_NU} rad/s, lambda = {SHAFT_LAM} 1/s, the acoustic bench fit of "
        "`results/noise_v2/shaft/findings.md`) plus a white floor; control (ii) is that same "
        f"render plus an independent per-order Wiener with `D_k = {CTRL_D1} k` rad^2/s, and is "
        "corrected with control (i) as its own floor, so the pair is internally consistent.\n"
    )
    null = res["null_check"]
    lines.append(
        f"Both controls are run TWICE: once on the shaft each recording actually has "
        f"(`sigma_nu` fitted per recording from its own measured `V_shared` at fixed "
        f"`lambda = {SHAFT_LAM}`, which comes out at {np.min(sig):.2f} - {np.max(sig):.2f} rad/s, "
        f"median {np.median(sig):.2f}), and once on the literally mandated "
        f"`sigma_nu = {SHAFT_SIGMA_NU}`. The MATCHED pair is the headline and the matched floor is "
        "what is subtracted from the data, because the published bench fit absorbed the per-order "
        "decoherence into the shared term and so over-states the real shared phase noise by about "
        "a factor three - and a control whose shaft is three times too large is not a valid "
        "floor: its low-order lines are broader, its unwrapped low-order phase slips more, and "
        "its shaft-estimate error comes out several times larger, which above order 20 drives the "
        "leakage correction past the observed residual. The nominal pair is reported alongside, "
        "and its behaviour is itself the evidence for that statement.\n"
    )
    lines.append(
        f"- **Control (i) returns the floor.** On the matched shaft its `V_eps` is flat in `tau` "
        f"to within {rec['floor_flatness']:.0%} between {FLAT_LAG_LO_MS:.0f} and "
        f"{FIT_TAU_HI_MS:.0f} ms - it carries no `tau` dependence at all, which is what an "
        f"additive-noise floor must look like - and its level is {rec['floor_vs_analytic']:.2f} "
        "times the analytic phase floor `2 log(1 + 1/SNR)` of the measured line-to-noise ratio, "
        f"cell by cell. Below {FLAT_LAG_LO_MS:.0f} ms the floor is genuinely smaller, because the "
        "noise inside the band is correlated over about 1/(2 band) = 16 ms and the two samples of "
        "a short increment share part of it. No independent term is invented where none exists. "
        f"The floor's own log-log slope against `k` is {rec['floor_k_slope']:+.2f} - no `k^2` "
        "shape, so the leakage correction has not left a shaft-shaped term behind for the "
        f"subtraction to remove from the data. On the nominal shaft the same three read-outs are "
        f"{rec_nom['floor_flatness']:.0%}, {rec_nom['floor_vs_analytic']:.2f} and "
        f"{rec_nom['floor_k_slope']:+.2f}, and its floor goes NEGATIVE at "
        f"{rec_nom['floor_n_negative_orders']} of {rec_nom['floor_n_orders']} orders against "
        f"{rec['floor_n_negative_orders']} on the matched shaft - that is the out-of-range "
        "behaviour, in one number.\n"
        f"- **Control (ii) is recovered.** Over the {rec['n_cells']} resolvable cells of the fit "
        f"domain the recovered `V_eps` is {rec['ratio_median']:.2f} times the planted "
        f"`2 D_k tau` (interquartile {rec['ratio_q25']:.2f} - {rec['ratio_q75']:.2f}), and the "
        f"power-law fit of the recovered values gives a = {rec['fit']['a']:.4g} "
        f"(planted {2 * CTRL_D1:.2f}), p = {rec['fit']['p']:.3f} (planted 1), "
        f"q = {rec['fit']['q']:.3f} (planted 1). On the nominal shaft the recovery ratio is "
        f"{rec_nom['ratio_median']:.2f} over {rec_nom['n_cells']} cells.\n"
        f"- **The subtraction returns zero when there is nothing to find.** A SECOND matched "
        f"floor render - same shaft amplitude and same per-cell SNR, independent noise, blade "
        f"phases and shaft realisation - put through the same subtraction against the first gives "
        f"{null['n_resolvable']} resolvable cells over the fit domain against "
        f"{res['fit']['net']['n_cells']} for the bench, and a median |net| of "
        f"{null['abs_median']:.4g} rad^2 against {null['data_abs_median']:.4g} rad^2. The "
        "residual the bench shows is not the subtraction's own noise.\n"
    )
    lines.append(markdown_table(control_table(res)) + "\n")
    lines.append(
        "Control (ii) plants a per-order Wiener that is SHARED across microphones, so its "
        f"cross-microphone `rho` is high ({rec['mic_planted']:.2f} over k = 4-24) while control "
        f"(i), whose only per-order term is independent sensor noise, gives "
        f"{rec['mic_floor']:.2f}. The two controls bracket the cross-microphone test, and the "
        "measured bench value is read against them.\n"
    )

    lines.append("## What this means\n")
    lines.append(res["verdict_paragraph"] + "\n")

    lines.append("## How the numbers were made\n")
    lines.append(
        "- **Window.** Eleven of the twelve files are 25 s long and the motor runs for only about "
        "11 s of that; `motor_Motor1_70` runs for 31 s. The window is the longest stretch of the "
        f"level track within {LEVEL_DROP_DB:.0f} dB of its 90th percentile, trimmed by "
        f"{ACTIVE_EDGE_S} s at each end and capped at {MAX_SEG_S:.0f} s. A fixed 24 s window from "
        "t = 3 s is half room noise, and the uniform phase of the silent half alone produces a "
        "linearly growing increment variance at every order.\n"
    )
    lines.append(
        f"- **Demodulation.** Constant carrier per recording, refined with "
        f"`residual_frequency` at orders {REFINE_ORDERS} so that every line up to {K_MAX} sits "
        f"within about 1 Hz of band centre; band `+-{BAND_FRAC} * rate`, decimated to "
        f"{FS_PHASE:.1f} Hz. Audio is anti-aliased and decimated by {PRE_DECIM} first "
        f"(Nyquist {FS_WORK / 2:.0f} Hz, the highest analysed line is 3.6 kHz); this reproduces "
        "the full-rate SNR of every cell to three decimals and makes the ladder four times "
        "cheaper.\n"
    )
    lines.append(
        f"- **Gate.** A cell is used when the peak of its `|z_k|^2` spectrum stands "
        f"{SNR_MIN_DB:.0f} dB above the median of the same spectrum inside the band; weights are "
        f"that ratio. An order enters the fit only if it clears that bar on at least "
        f"{ORDER_MIN_MICS} of the 8 microphones: a per-cell gate on its own selects the luckiest "
        "channel of a marginal order, whose SNR reads high because its own noise fluctuated low, "
        "and that under-states the cell's floor. The weak odd orders of this two-bladed rotor are "
        "what the bar removes.\n"
    )
    lines.append(
        "- **Why the variance is circular, not unwrapped.** In a `+-0.4 rate` band the "
        "line-to-noise power ratio of a high order on this bench is a few dB, so `np.unwrap` "
        "takes the wrong branch a few times per hundred samples. Each slip adds `(2 pi)^2` and "
        "the slip count random-walks, so the unwrapped variance grows linearly in `tau` whatever "
        "the phase does: it reads "
        f"{pooled[str(lag_b)]['v_unwrap'][0]:.0f} rad^2 at order 1 and {lag_b} ms, for a line "
        "0.1 Hz wide. The headline variance is `-2 log |gamma|` of the amplitude-weighted "
        "circular mean of the de-rotated increment phasor, which cannot slip. `v_unwrap` in the "
        "JSON is the mandated unwrapped estimator, kept so the size of the artefact is on "
        "record.\n"
    )
    lines.append(
        "- **Leakage.** `dtheta_hat` is a weighted sum of the same noisy increments, so its error "
        "`e` reappears in every residual as `k^2 Var[e]`. `Var[e]` is measured: the shaft band is "
        f"split into {N_GROUPS} disjoint order groups, the three independent shaft estimates are "
        "differenced pairwise, and `Var[e]` follows from the three pairwise variances. Orders "
        "inside the shaft band are de-rotated leave-group-out. The correction is "
        f"{res['leakage_share']:.0%} of the observed residual at order {FIT_K_HI} and "
        f"{lag_a} ms.\n"
    )
    lines.append(
        f"- **Leave-band-out.** The headline shaft estimate already uses only k <= {K_SHAFT}: "
        "above that the per-sample phase is noise-dominated and its unwrapped phase slips. The "
        "literal all-gated-orders estimate of step 4 is reported alongside as `v_all_orders` and "
        f"`v_shaft_all_orders`; it gives `V_shared({lag_a} ms) = "
        f"{a['v_shaft_all_orders']:.3e}` rad^2 against {a['v_shaft']:.3e} for the low band, "
        "because `k^2` in the normal equations hands most of the weight to orders whose phase is "
        "noise.\n"
    )
    lines.append(
        f"- **Ceiling.** `-2 log |gamma|` stops being informative once the residual has "
        f"decohered; cells with `V_obs > {V_CEIL}` rad^2 are reported but excluded from the fit. "
        "Control (ii) fixes that bar: the planted `D_k` is returned wherever `V_obs` is under it "
        "and is lost above it.\n"
    )
    lines.append("## Per recording\n")
    rows = [
        [
            "recording",
            "rate [rev/s]",
            "survey [rev/s]",
            "window [s]",
            "start [s]",
            "band [Hz]",
            "cells",
            f"V_shared({lag_a} ms)",
            f"V_eps(k=8, {lag_a} ms)",
            "median slip [%]",
        ]
    ]
    for r in res["recordings"]:
        e = r["orders"].get(str(lag_a), {})
        rows.append(
            [
                r["id"],
                _f(r["rate_rps"], "{:.3f}"),
                _f(r["rate_survey_rps"], "{:.2f}"),
                _f(r["seg_len_s"], "{:.1f}"),
                _f(r["seg_start_s"], "{:.1f}"),
                _f(r["band_hz"], "{:.1f}"),
                str(r["n_cells_total"]),
                _f(e.get("v_shaft"), "{:.3e}"),
                _f((e.get("v_net") or [float("nan")] * K_MAX)[7], "{:.3g}"),
                _f(100.0 * r["slip_median"], "{:.2f}"),
            ]
        )
    lines.append(markdown_table(rows) + "\n")

    path = out_dir / "findings.md"
    path.write_text("\n".join(lines))
    return path


def long_order_rows(blob: dict[str, Any], lags: tuple[int, ...]) -> list[list[str]]:
    """One table of a long-lag source: per order, the curve, both fits, the verdict."""
    head = (
        ["k"]
        + [f"V({ms / 1000:g} s)" for ms in lags]
        + ["n fit", "q", "a", "tau_c [s]", "c", "dAIC", "verdict"]
    )
    rows = [head]
    for k in sorted((int(k) for k in blob), key=int):
        cell = blob[str(k)]
        fit = cell["fit"]
        power = fit.get("power", {})
        sat = fit.get("saturating", {})
        # A lag whose OBSERVED residual is over the estimator ceiling carries
        # no shape and is excluded from the fit; its value is still tabulated,
        # in brackets, because the bend it shows is the instrument's.
        cells = [
            f"({_f(cell['v_table'].get(str(ms)))})"
            if cell["ceil_table"].get(str(ms))
            else _f(cell["v_table"].get(str(ms)))
            for ms in lags
        ]
        rows.append(
            [
                f"{k}" + ("*" if cell["at_ceiling"] else ""),
                *cells,
                _f(fit.get("n_points"), "{:d}"),
                _f(power.get("q"), "{:+.2f}"),
                _f(power.get("a")),
                _f(sat.get("tau_c"), "{:.2f}"),
                _f(sat.get("c")),
                _f(fit.get("d_aic"), "{:+.1f}"),
                fit.get("verdict", "-"),
            ]
        )
    return rows


def write_long_findings(res: dict[str, Any], out_dir: Path) -> Path:
    """Findings of the long-lag extension: numbers first, then the verdict."""
    long = res["long_lag"]
    lags = tuple(ms for ms in LONG_TABLE_LAGS_MS if ms in long["lags_ms"])
    src = long["sources"]
    lines: list[str] = []
    lines.append("# Order decoherence at long lags - findings\n")
    lines.append(
        "Source: `scripts/noise_v2_order_decoherence.py --long --max-lag-s 5`. Numbers: "
        "`results/noise_v2/decoherence_long/decoherence_long.json`. Figure: "
        "`docs/explainers/noise-model-v2-plan/decoherence_long.png`.\n"
    )
    lines.append(
        "The 0.5 s study returned an independent per-order term that grows as "
        "`V_eps(k, tau) = a k^p tau^q` with `q` near 1 - a Wiener phase per order, which has NO "
        "ceiling. A per-order OU PHASE has one: `V_eps` saturates at `2 sigma_eps^2/lambda_eps` "
        "after `tau_c = 1/lambda_eps`, and below `tau_c` the two shapes are the same curve. The "
        "question is therefore only decidable at lags long enough to reach `tau_c`, which is what "
        "this run measures: the same estimator, the same controls, lags out to "
        f"{max(long['lags_ms']) / 1000:g} s on the windows long enough to hold "
        f"{LAG_WINDOW_FRAC:.0f} of them.\n"
    )

    lines.append("## The supports\n")
    rows = [["support", "window [s]", "from [s]", "rate [rev/s]", "cells", "lags [s]", "window"]]
    for r in res["recordings"]:
        got = sorted(int(ms) for ms in r["orders"])
        rows.append(
            [
                f"`{r['id']}`",
                _f(r["seg_len_s"], "{:.1f}"),
                _f(r["seg_start_s"], "{:.1f}"),
                _f(r["rate_rps"], "{:.3f}"),
                str(r["n_cells_total"]),
                ", ".join(f"{ms / 1000:g}" for ms in got if ms >= LONG_CEIL_LAG_MS),
                str(r.get("window_source")),
            ]
        )
    lines.append(markdown_table(rows) + "\n")
    lines.append(
        f"`{LONG_LEVEL_RECORDING}` enters twice: once on the level-detected active run - the "
        "longest continuous single-rotor window this project holds - and once on its published "
        "`noise-v2-bench-points` window. The other two supports are the remaining single-rotor "
        f"points whose published span reaches {LONG_SPAN_MIN_S:.0f} s. Orders are the "
        f"{', '.join(str(k) for k in K_FAMILY)} family, all of them below the "
        f"{AA_CORNER_HZ:.0f} Hz anti-alias corner at these rates "
        f"(order {K_FAMILY[-1]} of the fastest support sits at "
        f"{K_FAMILY[-1] * max(r['rate_rps'] for r in res['recordings']):.0f} Hz).\n"
    )

    lines.append("## The numbers\n")
    lines.append(
        "`V` is `V_eps` in rad^2: observed, minus the measured leakage of the shaft estimate, "
        "minus the additive-noise floor of that support's paired control. `q` and `a` are the "
        "power law `V = a tau^q`, `tau_c` and `c` the saturating `V = c (1 - exp(-tau/tau_c))`, "
        f"both fitted by least squares on `log V` over the lags from "
        f"{long['fit_tau_lo_ms'] / 1000:g} s up; `n fit` is how many lags that left. "
        "`dAIC = AIC(power) - AIC(saturating)`, so a POSITIVE value favours the ceiling; the "
        f"decision bar is {AIC_DECISIVE:.0f}.\n"
    )
    lines.append(
        f"Two read-out limits are marked. A bracketed `V` is a lag whose OBSERVED residual is "
        f"over the {V_CEIL:.0f} rad^2 estimator ceiling: there `|gamma|` has reached its own "
        "sampling floor, the curve bends because the instrument has run out, and the lag takes "
        "part in NO fit (control (ii) is what proves this - its planted random walk crosses the "
        "ceiling at the high orders and the raw curve then flattens). A `*` on `k` marks an order "
        f"already over the ceiling at {LONG_CEIL_LAG_MS} ms, i.e. one that has no readable long "
        f"lag at all. A `tau_c` beyond the longest FITTED lag means the saturating curve has "
        "straightened into the power law inside the data, which is not evidence of a ceiling; "
        f"a `verdict` of `flat` means the curve grows by less than {FLAT_GROWTH_RATIO:.2f} across "
        "the fitted lags, so there is no lag dependence left for either shape to explain.\n"
    )
    n_sup = len(res["recordings"])
    lines.append(f"### Pooled over the {n_sup} long support{'s' if n_sup != 1 else ''}\n")
    lines.append(markdown_table(long_order_rows(src["data"]["pooled"], lags)) + "\n")
    for sup in src["data"]["per_support"]:
        lines.append(
            f"### `{sup['id']}` ({sup['seg_len_s']:.1f} s, {sup['rate_rps']:.3f} rev/s, "
            f"{sup['window_source']} window)\n"
        )
        lines.append(markdown_table(long_order_rows(sup["orders"], lags)) + "\n")

    lines.append("## The controls at the same lags\n")
    lines.append(
        "Control (i) is the paired floor render - the same rate, the same per-(order, microphone) "
        "line-to-noise ratio, a shared integrated-OU shaft and nothing else - so the quantity "
        "tabulated for it is its own leakage-corrected residual, i.e. the additive-noise floor, "
        "which must be FLAT in `tau` (`q` near 0). Control (ii) is that render plus a planted "
        f"independent per-order Wiener phase, `D_k = {CTRL_D1} k` rad^2/s, whose net term must "
        "keep growing as `2 D_k tau` (`q` near 1). They calibrate what a plateau and what "
        "continued growth look like through this estimator at these lags.\n"
    )
    for name, title in (
        ("matched", "Control (i): shaft + floor only (the floor; must be flat)"),
        ("matched_planted", f"Control (ii): planted D_k = {CTRL_D1} k (must keep growing)"),
    ):
        lines.append(f"### {title}\n")
        lines.append(markdown_table(long_order_rows(src[name]["pooled"], lags)) + "\n")
    trust = long["ceiling_trust_plateau"]
    lines.append(
        f"Control (ii) is also what sets the level above which a saturating fit means "
        f"nothing. It has no ceiling by construction, yet the same comparison fits it one at "
        f"a plateau of {trust:.3g} rad^2 (its verdict row `ceiling`): as `V` approaches the "
        f"{V_CEIL:.0f} rad^2 read-out limit the circular mean compresses, and the curve bends "
        f"before the lag mask removes it. Every data order whose fitted plateau reaches "
        f"{trust:.3g} rad^2 is therefore reported as `undetermined (plateau where control "
        "(ii) fakes one)` rather than as a ceiling - this run cannot tell those two apart, "
        "and saying otherwise would be reading the instrument, not the rotor.\n"
        if trust is not None
        else "Control (ii) is never fitted a ceiling at these levels, so no data plateau had "
        "to be discounted as the estimator's own compression.\n"
    )

    lines.append("## The verdict\n")
    lines.append(long["verdict"] + "\n")
    lines.append(long["verdict_paragraph"] + "\n")

    lines.append("## What this run does NOT settle\n")
    lines.append(
        f"- The windows are {min(r['seg_len_s'] for r in res['recordings']):.0f}-"
        f"{max(r['seg_len_s'] for r in res['recordings']):.0f} s, so a {max(long['lags_ms']) / 1000:g} s "
        f"lag has only about {min(r['seg_len_s'] for r in res['recordings']) / (max(long['lags_ms']) / 1000):.0f} "
        "disjoint increments in it; the long end of every curve is the noisiest part of it.\n"
        "- All four supports are ONE rotor of ONE rig (DREGON Motor 1 at 50, 60 and 70 % "
        "throttle). A ceiling or its absence here is a statement about this rotor, not about "
        "every rig in the corpus.\n"
        "- The high orders are at the estimator ceiling long before 1 s, so the shape can only "
        "ever be read on the low orders. Nothing here contradicts or confirms the high-order "
        "behaviour.\n"
        "- `V_eps` is a floor-subtracted difference of two circular-estimator read-outs; at the "
        "long lags the two terms are close, so its relative error is larger than the short-lag "
        "one. The `net_err` column of the JSON carries the standard error of each cell.\n"
    )

    path = out_dir / "findings.md"
    path.write_text("\n".join(lines))
    return path


# ═════════════════════════════════════════════════════════════════════════════
# Verdicts
# ═════════════════════════════════════════════════════════════════════════════


def verdicts(res: dict[str, Any]) -> None:
    blocks = res["correlations"]["blocks"]
    mic = np.asarray(res["correlations"]["mic_rho"], dtype=np.float64)
    mic_mid = float(np.nanmean(mic[MID_K]))
    off = blocks["off_diagonal_abs"]
    parity_gap = abs(blocks["same_parity"] - blocks["cross_parity"])
    if off <= CORR_INDEP_MAX and parity_gap <= CORR_INDEP_MAX:
        order_v = (
            f"independent across orders: mean |rho| = {off:.3f} is inside the {CORR_INDEP_MAX:.2f} "
            "bar and the odd/even blocks do not separate, so one Wiener phase per order with no "
            "cross-order structure is the right generator."
        )
    elif parity_gap > CORR_INDEP_MAX:
        order_v = (
            f"block-structured: same-parity pairs correlate at {blocks['same_parity']:.3f} against "
            f"{blocks['cross_parity']:.3f} across parity, a gap of {parity_gap:.3f}, so the "
            "two-bladed rotor's odd and even families share a phase term."
        )
    else:
        order_v = (
            f"correlated across orders: mean |rho| = {off:.3f} exceeds the "
            f"{CORR_INDEP_MAX:.2f} bar without an odd/even split, so the residual carries a "
            "further common term that the k-proportional model does not span."
        )
    res["correlations"]["order_verdict"] = order_v

    if mic_mid >= CORR_SHARED_MIN:
        mic_v = (
            f"source-side: the residual of one order is shared across microphones at rho = "
            f"{mic_mid:.3f}, so it is a property of the rotor, not of the propagation path, and "
            "one per-order phase process serves every channel."
        )
    elif mic_mid <= CORR_INDEP_MAX:
        mic_v = (
            f"path-side: the residual of one order is independent across microphones "
            f"(rho = {mic_mid:.3f}), so it is generated between the rotor and each sensor and a "
            "renderer needs one process per channel."
        )
    else:
        mic_v = (
            f"mixed: cross-microphone rho = {mic_mid:.3f} sits between the independent bar "
            f"({CORR_INDEP_MAX:.2f}) and the shared bar ({CORR_SHARED_MIN:.2f}), so part of the "
            "per-order phase noise is source-side and part is path-side."
        )
    res["correlations"]["mic_verdict"] = mic_v

    lag_a, lag_b = TABLE_LAGS_MS
    pooled = res["pooled"]
    a, b = pooled[str(lag_a)], pooled[str(lag_b)]
    fit = res["fit"]["net"]
    boot = res["fit"]["net_bootstrap"]
    par = res["parity"]
    net_a = np.asarray(a["v_net"], dtype=np.float64)
    net_b = np.asarray(b["v_net"], dtype=np.float64)
    ratio_a = net_a / (a["v_shaft"] * np.arange(1, K_MAX + 1) ** 2)
    finite = np.isfinite(ratio_a)
    k_cross = None
    for k in range(FIT_K_LO, K_MAX + 1):
        if finite[k - 1] and ratio_a[k - 1] < 1.0:
            k_cross = k
            break
    para = (
        f"The independent per-order term is not a correction, it is the dominant part of the "
        f"harmonic phase noise on this bench. At {lag_a} ms the tied model predicts "
        f"`k^2 V_shared = {a['v_shaft'] * 64:.3g}` rad^2 at order 8 and "
        f"{a['v_shaft'] * 1600:.3g} rad^2 at order 40, while the measured independent term is "
        f"{net_a[7]:.3g} and {net_a[39]:.3g} rad^2 - a factor {ratio_a[7]:.0f} and "
        f"{ratio_a[39]:.0f} more than the shared term carries"
        + (
            f", and the shared term only catches up above order {k_cross}. "
            if k_cross
            else ", at every order the gate resolves. "
        )
        + f"It does scale with the order, but far more slowly than the `k^2` of a tied phase: "
        f"`V_eps = {fit['a']:.3g} k^{fit['p']:.2f} tau^{fit['q']:.2f}` rad^2, with "
        f"p = {fit['p']:.2f} (95 % CI {boot['p']['lo95']:.2f} - {boot['p']['hi95']:.2f}) and "
        f"q = {fit['q']:.2f} (95 % CI {boot['q']['lo95']:.2f} - {boot['q']['hi95']:.2f}), i.e. "
        f"close to a Wiener phase per order with a diffusion roughly proportional to `k`, "
        f"`D_k = {0.5 * fit['a']:.3g} k^{fit['p']:.2f}` rad^2/s. At {lag_b} ms the term has grown "
        f"to {net_b[7]:.3g} rad^2 at order 8 and {net_b[39]:.3g} rad^2 at order 40, which is "
        f"decoherence, not jitter: a harmonic that has lost {net_b[39]:.3g} rad^2 of phase over "
        f"half a second cannot be rendered from the shaft label alone. The size is not one number "
        f"per order either: the odd orders of this two-bladed rotor carry "
        f"{par[str(lag_a)]['ratio']:.1f} times the independent phase noise of the even orders, so "
        f"a generator needs the weak, dissimilarity-driven family to be noisier than the "
        f"blade-passing family. "
    )
    if mic_mid >= CORR_SHARED_MIN:
        para += (
            f"It is source-side: the same residual appears on all eight microphones "
            f"(rho = {mic_mid:.2f}), so it belongs to the rotor and a renderer needs one extra "
            "phase process per order, not per channel."
        )
    elif mic_mid <= CORR_INDEP_MAX:
        para += (
            f"It is path-side: the residual is independent across microphones (rho = "
            f"{mic_mid:.2f}), so it is generated in propagation and a renderer needs one process "
            "per order AND per channel."
        )
    else:
        para += (
            f"It is partly source-side and partly path-side: cross-microphone rho = "
            f"{mic_mid:.2f}, so about that fraction of the per-order variance belongs to the "
            "rotor and the rest to each path."
        )
    res["verdict_paragraph"] = para


def mic_rho_mean(rows: list[dict[str, Any]]) -> float:
    """Mean cross-microphone correlation of the residual over :data:`MID_K`."""
    return float(
        np.nanmean(
            np.stack([np.asarray(r["correlations"]["mic_rho"], np.float64)[MID_K] for r in rows])
        )
    )


def null_check(null_rows: list[dict[str, Any]], rec_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What the floor subtraction returns when there is nothing to find.

    ``null_rows`` is a SECOND floor render - same shaft amplitude, same
    per-cell line-to-noise ratio, independent noise, blade phases and shaft
    realisation - corrected with the first. Its net term must be sampling
    error around zero. Anything the bench shows above this is real.
    """
    out: dict[str, Any] = {}
    band = slice(FIT_K_LO - 1, FIT_K_HI)
    for name, rows in (("", null_rows), ("data_", rec_rows)):
        vals: list[float] = []
        n_res = 0
        for ms in LAGS_MS:
            if not (FIT_TAU_LO_MS <= ms <= FIT_TAU_HI_MS):
                continue
            net = np.nanmean(stack_field(rows, ms, "v_net"), axis=0)[band]
            res = consensus(rows, ms)[band]
            vals.extend(net[np.isfinite(net)].tolist())
            n_res += int(res.sum())
        arr = np.abs(np.asarray(vals, dtype=np.float64))
        out[f"{name}abs_median"] = float(np.median(arr)) if arr.size else float("nan")
        out[f"{name}n_resolvable"] = n_res
    return out


def control_recovery(
    floor_rows: list[dict[str, Any]],
    plant_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Control read-outs: is the floor flat and at the analytic level, is ``D_k`` returned?"""
    fl = {ms: np.nanmean(stack_field(floor_rows, ms, "v_leak_corrected"), axis=0) for ms in LAGS_MS}
    fl_an = {
        ms: np.nanmean(stack_field(floor_rows, ms, "v_floor_analytic"), axis=0) for ms in LAGS_MS
    }
    # The additive-noise phase error is correlated over about 1/(2 band) = 16
    # ms, so the floor is only expected to be flat once the two samples of the
    # increment are independent. Flatness is therefore read from FLAT_LAG_LO_MS
    # up, and the shorter lags are reported as they are.
    inband = [ms for ms in LAGS_MS if FLAT_LAG_LO_MS <= ms <= FIT_TAU_HI_MS]
    arr = np.stack([fl[ms] for ms in inband])
    k_ok = np.arange(FIT_K_LO - 1, K_MAX)
    with np.errstate(invalid="ignore"):
        # Flatness is only defined on orders whose floor is positive at EVERY
        # in-band lag. Where the leakage correction has driven a floor through
        # zero the ratio is meaningless, and those orders are counted instead.
        col = arr[:, k_ok]
        usable = np.all(np.isfinite(col) & (col > 0), axis=0)
        spread = np.nanmax(col[:, usable], axis=0) / np.nanmin(col[:, usable], axis=0)
        flatness = float(np.median(spread) - 1.0) if usable.any() else float("nan")
        n_negative = int((~usable).sum())
        vs_analytic = float(
            np.nanmedian(
                np.stack([fl[ms][k_ok][usable] for ms in inband])
                / np.maximum(np.stack([fl_an[ms][k_ok][usable] for ms in inband]), 1e-12)
            )
        )
        # Does the measured floor carry a spurious k^2 shape? A shaft-estimate
        # leakage left uncorrected would, and it would then be subtracted off
        # the data and flatten the measured k exponent. The log-log slope of
        # the floor against k over the fit band answers it.
        ks_fit = np.arange(FIT_K_LO, K_MAX + 1, dtype=np.float64)
        lk, lv = [], []
        for ms in inband:
            col = fl[ms][k_ok]
            good = np.isfinite(col) & (col > 0)
            lk.append(np.log(ks_fit[good]))
            lv.append(np.log(col[good]))
        lk_all, lv_all = np.concatenate(lk), np.concatenate(lv)
        floor_k_slope = float(
            np.linalg.lstsq(np.stack([np.ones_like(lk_all), lk_all], axis=1), lv_all, rcond=None)[
                0
            ][1]
        )
    cells: list[dict[str, Any]] = []
    ratios: list[float] = []
    for ms in LAGS_MS:
        net = np.nanmean(stack_field(plant_rows, ms, "v_net"), axis=0)
        obs = np.nanmean(stack_field(plant_rows, ms, "v_obs"), axis=0)
        tau = ms * 1e-3
        for idx in range(K_MAX):
            k = idx + 1
            planted = 2.0 * CTRL_D1 * k * tau
            good = (
                FIT_K_LO <= k <= FIT_K_HI
                and FIT_TAU_LO_MS <= ms <= FIT_TAU_HI_MS
                and np.isfinite(obs[idx])
                and obs[idx] <= V_CEIL
                and np.isfinite(net[idx])
                and net[idx] > 0
            )
            if good:
                ratios.append(net[idx] / planted)
            if k in K_FAMILY and ms in TABLE_LAGS_MS:
                cells.append(
                    {
                        "k": k,
                        "lag_ms": ms,
                        "planted": planted,
                        "recovered": float(net[idx]) if np.isfinite(net[idx]) else None,
                        "ratio": float(net[idx] / planted) if np.isfinite(net[idx]) else None,
                        "floor": float(fl[ms][idx]) if np.isfinite(fl[ms][idx]) else None,
                        "floor_analytic": float(fl_an[ms][idx])
                        if np.isfinite(fl_an[ms][idx])
                        else None,
                        "resolvable": bool(good),
                    }
                )
    fit = fit_from_rows(plant_rows, "v_net") or {
        "a": float("nan"),
        "p": float("nan"),
        "q": float("nan"),
    }
    r = np.asarray(ratios, dtype=np.float64)

    return {
        "floor_flatness": flatness,
        "floor_vs_analytic": vs_analytic,
        "floor_k_slope": floor_k_slope,
        "floor_n_negative_orders": n_negative,
        "floor_n_orders": int(K_MAX - FIT_K_LO + 1),
        "n_cells": int(r.size),
        "ratio_median": float(np.median(r)) if r.size else float("nan"),
        "ratio_q25": float(np.percentile(r, 25)) if r.size else float("nan"),
        "ratio_q75": float(np.percentile(r, 75)) if r.size else float("nan"),
        "fit": fit,
        "cells": cells,
        "mic_floor": mic_rho_mean(floor_rows),
        "mic_planted": mic_rho_mean(plant_rows),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Driver
# ═════════════════════════════════════════════════════════════════════════════


def pool_correlations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mats = np.stack([np.asarray(r["correlations"]["order_corr"], dtype=np.float64) for r in rows])
    use = np.stack([np.asarray(r["correlations"]["order_usable"], dtype=bool) for r in rows])
    with np.errstate(invalid="ignore"):
        corr = np.nanmean(mats, axis=0)
    usable = use.sum(axis=0) >= max(1, len(rows) // 2)
    mic = np.nanmean(
        np.stack([np.asarray(r["correlations"]["mic_rho"], dtype=np.float64) for r in rows]), axis=0
    )
    return {
        "lag_ms": CORR_LAG_MS,
        "order_corr": corr.tolist(),
        "order_usable": usable.tolist(),
        "mic_rho": mic.tolist(),
        "blocks": block_reads(corr, usable),
    }


def admitted_lags(max_lag_s: float) -> tuple[int, ...]:
    """The published ladder plus the long-lag entries ``max_lag_s`` admits."""
    return LAGS_MS + tuple(ms for ms in LAGS_MS_LONG if ms <= max_lag_s * 1000.0 + 1e-9)


def run(
    limit: int | None,
    out_dir: Path,
    fig_dir: Path,
    *,
    long_mode: bool = False,
    max_lag_s: float = 0.5,
) -> dict[str, Any]:
    lags_ms = admitted_lags(max_lag_s)
    rates = survey_rates()
    bench = load_long(limit) if long_mode else load_bench(limit)
    if not bench:
        raise RuntimeError("no DREGON bench recording found in DREGON-frames")
    print(f"{len(bench)} bench recordings, lags {[ms for ms in lags_ms]} ms")

    # Each list holds one row per recording. The reference in CTRL_REFERENCE is
    # the support whose residual is subtracted as that row's floor.
    groups: dict[str, list[dict[str, Any]]] = {name: [] for name in ("data", *CTRL_REFERENCE)}
    for job in bench:
        rid = job.name
        seg, start_s, len_s = job_segment(job)
        work = pre_decimate(seg)
        rate0 = job.rate_seed or rates.get(job.meta.get("recording_id", rid))
        if rate0:
            rate = rate0
        else:
            # The survey has every bench recording; fall back on DREGON's own
            # throttle law only if that ever changes.
            m = BENCH_RE.match(rid)
            if m is None:
                raise RuntimeError(f"{rid} does not look like a bench recording")
            rate = 0.975 * float(m.group(2)) + 0.37
        for order in REFINE_ORDERS:
            rate, _ = refine_rate(work, rate, order)
        z = demod_ladder(work, rate)
        sup = analyse_support(rid, z, rate, BAND_FRAC * rate, lags_ms)
        sup.seg_start_s, sup.seg_len_s, sup.rate_survey = start_s, len_s, rate0
        ctrl, prov = build_controls(sup, work.shape[-1], lags_ms)
        ctrl["data"] = sup

        for name, ref_name in (("data", "matched"), *CTRL_REFERENCE.items()):
            sp = ctrl[name]
            ref = ctrl[ref_name] if ref_name else None
            groups[name].append(
                {
                    "id": sp.name,
                    "rate_rps": sp.rate,
                    "rate_survey_rps": sup.rate_survey,
                    "band_hz": sp.band,
                    "seg_start_s": sup.seg_start_s,
                    "seg_len_s": sup.seg_len_s,
                    "n_frames": sp.n_frames,
                    "snr_pm_db": (10.0 * np.log10(sp.snr_pm)).round(2).tolist(),
                    "snr_eff_db": (10.0 * np.log10(sp.snr_eff)).round(2).tolist(),
                    "n_cells_total": int(sp.keep.sum()),
                    "slip_median": float(np.median(sp.slip)),
                    "window_source": job.window_source,
                    "support_meta": job.meta,
                    "control_provenance": prov,
                    "orders": reduce_support(sp, ref),
                    "correlations": correlation_reads(sp),
                }
            )
        e = groups["data"][-1]["orders"][str(TABLE_LAGS_MS[0])]
        print(
            f"  {rid}: rate {rate:.3f} rev/s, window {len_s:.1f} s from {start_s:.1f} s, "
            f"{int(sup.keep.sum())} cells, sigma_nu {prov['sigma_matched']:.2f} rad/s, "
            f"V_shared(50 ms) {e['v_shaft']:.3e}, V_eps(k=8) {e['v_net'][7]:.3g}"
        )
    rec_rows = groups["data"]

    res: dict[str, Any] = {
        "config": {
            "fs_native_hz": FS_NATIVE,
            "fs_work_hz": FS_WORK,
            "fs_phase_hz": FS_PHASE,
            "k_max": K_MAX,
            "band_frac": BAND_FRAC,
            "refine_orders": list(REFINE_ORDERS),
            "lags_ms": list(lags_ms),
            "max_lag_s": max_lag_s,
            "long_support_set": long_mode,
            "lag_window_frac": LAG_WINDOW_FRAC,
            "long_fit_tau_lo_ms": LONG_FIT_TAU_LO_MS,
            "snr_min_db": SNR_MIN_DB,
            "k_shaft": K_SHAFT,
            "n_groups": N_GROUPS,
            "slip_sigma": SLIP_SIGMA,
            "v_ceil": V_CEIL,
            "fit_domain": {
                "k": [FIT_K_LO, FIT_K_HI],
                "tau_ms": [FIT_TAU_LO_MS, FIT_TAU_HI_MS],
            },
            "control": {"sigma_nu": SHAFT_SIGMA_NU, "lam": SHAFT_LAM, "D1": CTRL_D1},
            "n_boot": N_BOOT,
        },
        "n_recordings": len(rec_rows),
        "recordings": rec_rows,
        "controls": {name: groups[name] for name in CTRL_REFERENCE},
        "pooled": pooled_curves(rec_rows, lags_ms),
        "correlations": pool_correlations(rec_rows),
    }
    for name in CTRL_REFERENCE:
        res[f"pooled_control_{name}"] = pooled_curves(groups[name], lags_ms)
    res["correlations"]["mic_rho_controls"] = {
        name: pool_correlations(groups[name])["mic_rho"] for name in CTRL_REFERENCE
    }
    res["fit"] = {
        "net": fit_from_rows(rec_rows, "v_net"),
        "obs": fit_from_rows(rec_rows, "v_obs"),
        "net_bootstrap": bootstrap_fit(rec_rows, "v_net"),
        "net_even": fit_from_rows(rec_rows, "v_net", "even"),
        "net_even_bootstrap": bootstrap_fit(rec_rows, "v_net", "even"),
        "net_odd": fit_from_rows(rec_rows, "v_net", "odd"),
        "net_odd_bootstrap": bootstrap_fit(rec_rows, "v_net", "odd"),
    }
    res["control_recovery"] = {
        "matched": control_recovery(groups["matched"], groups["matched_planted"]),
        "nominal": control_recovery(groups["nominal"], groups["nominal_planted"]),
    }
    res["null_check"] = null_check(groups["matched_b"], rec_rows)
    lag_a = TABLE_LAGS_MS[0]
    leak = np.nanmean(stack_field(rec_rows, lag_a, "leakage"), axis=0)
    obs = np.nanmean(stack_field(rec_rows, lag_a, "v_obs"), axis=0)
    res["leakage_share"] = float(leak[FIT_K_HI - 1] / max(obs[FIT_K_HI - 1], 1e-12))
    res["parity"] = parity_reads(res["pooled"])
    if res["fit"]["net"]:
        verdicts(res)
    if any(ms in lags_ms for ms in LAGS_MS_LONG):
        res["long_lag"] = long_lag_reads(groups, lags_ms)

    out_dir.mkdir(parents=True, exist_ok=True)
    name = "decoherence_long.json" if long_mode else "decoherence.json"
    (out_dir / name).write_text(json.dumps(res, indent=1, allow_nan=True))
    print(f"  wrote {_rel(out_dir / name)}")
    if long_mode:
        fig_long(res, fig_dir / "decoherence_long.png")
        path = write_long_findings(res, out_dir)
    else:
        fig_structure(res, fig_dir / "decoherence_structure.png")
        fig_vs_k(res, fig_dir / "decoherence_vs_k.png")
        fig_corr(res, fig_dir / "decoherence_corr.png")
        path = write_findings(res, out_dir)
    print(f"  wrote {_rel(path)}")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Shared versus independent harmonic phase noise on the DREGON single-motor bench.",
    )
    ap.add_argument("--limit", type=int, default=None, help="analyse only the first N recordings")
    ap.add_argument(
        "--out",
        "--out-dir",
        dest="out",
        type=Path,
        default=None,
        help=f"results directory (default {_rel(DEFAULT_OUT)}, {_rel(DEFAULT_OUT_LONG)} with --long)",
    )
    ap.add_argument("--figs", type=Path, default=DEFAULT_FIGS, help="figure directory")
    ap.add_argument(
        "--long",
        action="store_true",
        help="the long-window support set: the level-detected active run of "
        f"{LONG_LEVEL_RECORDING} plus every single-rotor noise-v2-bench-points point whose "
        f"published span reaches {LONG_SPAN_MIN_S:.0f} s. Writes decoherence_long.json, "
        "findings.md and decoherence_long.png, and answers the long-lag shape question. "
        f"Implies --max-lag-s {LONG_MAX_LAG_S:g}.",
    )
    ap.add_argument(
        "--max-lag-s",
        type=float,
        default=None,
        help=f"longest lag admitted from the long extension {LAGS_MS_LONG} (seconds); the "
        "published ladder out to 1 s always runs. Default 0.5, i.e. the published ladder alone; "
        f"--long defaults it to {LONG_MAX_LAG_S:g}",
    )
    args = ap.parse_args(argv)
    out = args.out or (DEFAULT_OUT_LONG if args.long else DEFAULT_OUT)
    max_lag_s = (
        args.max_lag_s if args.max_lag_s is not None else (LONG_MAX_LAG_S if args.long else 0.5)
    )
    res = run(args.limit, out, args.figs, long_mode=args.long, max_lag_s=max_lag_s)
    if "long_lag" in res:
        print("\n" + res["long_lag"]["verdict"])
    fit = res["fit"]["net"]
    rec = res["control_recovery"]["matched"]
    if fit:
        print(f"\nV_eps = {fit['a']:.4g} k^{fit['p']:.3f} tau^{fit['q']:.3f}")
    print(
        f"control (i) floor: flat to {rec['floor_flatness']:.0%}, k-slope {rec['floor_k_slope']:+.2f}"
    )
    print(f"control (ii) recovery ratio median {rec['ratio_median']:.2f}")
    print(f"null subtraction: {res['null_check']['n_resolvable']} resolvable cells")
    if "order_verdict" in res["correlations"]:
        print(f"cross-order: {res['correlations']['order_verdict']}")
        print(f"cross-mic:   {res['correlations']['mic_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
