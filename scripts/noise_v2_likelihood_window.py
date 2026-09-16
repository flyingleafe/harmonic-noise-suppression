"""Noise-model-v2 criteria study A: the held-out spectral likelihood and the
analysis window a line-shape measurement can justify.

TWO measurements, one script, both on HELD-OUT material only.

1. **Line shape against the analysis window.** Each harmonic ``k`` of each
   rotor ``r`` is demodulated with its own supplied carrier -- the exact
   integrated telemetry phase ``2 pi k int f_r(t) dt`` -- so the modelled
   residual ``k theta_r + eps_rk + alpha`` is all that is left at baseband.
   The two-sided baseband PSD is then measured with Welch windows of
   ``T = 0.128 .. 4.096`` s (periodic Hann, 50 % overlap) and reported as

   * ``half_power_width_hz`` -- the baseline-subtracted line's half-power
     width, censored when the shape does not fall to half power inside the
     comb's own isolation limit;
   * ``frac_within_2bins`` -- the share of line power inside +-2 bins of the
     carrier, i.e. how much of the line a window of that length collects;
   * ``window_invariance_l1`` -- the L1 distance between the normalised line
     shape at ``T`` and at ``2T``, rebinned onto the coarser (``T``) grid.
     A window long enough to resolve the line makes this small.

   The baseband is NEVER widened past ``B = f_r / 2``, half the harmonic
   spacing: that is the widest band over which one harmonic of one rotor can
   be separated from its own neighbours, and a line wider than it is not a
   line at all. ``B`` travels with every number.

2. **Composite risk of three arms at three resolutions.** The frozen risk
   ``sum_i a_i [I_i / M_i + log M_i]`` on 30-7900 Hz, through the evaluator's
   own primitives (``revised_eval.marginal_frame_nll``, ``FrameScore``,
   ``composite_score``), for

   * ``oracle_np`` -- ``M`` is the Welch MEAN PERIODOGRAM of a DISJOINT real
     segment of the same recording at the same NFFT. A non-parametric oracle:
     no model, no carrier, only real material predicting real material. This
     is the floor every threshold is stated against;
   * ``current_best`` -- the baseline export the frozen evaluator uses
     (``read_export`` + ``aggregate_nuisance`` + ``predicted_m``, the
     historical-forward adapter);
   * ``c3`` -- the round-3 C3 revised export (``predict_spectrum``,
     ``mode='prior'``), the KNOWN-BAD arm (DREGON ``D = 844.69``).

   at ``NFFT in {2048, 4096, 16384}`` with ``hop = NFFT / 16``.

Also reported, so a window can be justified from the data rather than from
taste: the amplitude-envelope autocorrelation 1/e time of the demodulated
baseband, per rig and order -- the non-stationarity time scale the Welch
window must stay inside.

Nothing here is a gate and nothing here is fitted. Every number is a
measurement on held-out material, and every support is derived from the
frozen manifest by a stated rule.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy import signal

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.stochastic_fit import clips as C  # noqa: E402
from experiments.stochastic_fit import revised_eval as RE  # noqa: E402
from experiments.stochastic_fit import revised_phase as RP  # noqa: E402
from experiments.stochastic_fit.data import Clip, periodogram  # noqa: E402

TOPIC = "criteria"
OUT_DEFAULT = Path("results/noise_v2") / TOPIC
FIG_DEFAULT = Path("docs/explainers/noise-model-v2-plan")

SR = 16000
#: The hashed symmetric training guard of the frozen baseline manifest.
GUARD_S = 1.024
#: The frozen 30-7900 Hz observation band, read from the evaluator.
BAND_HZ = (RE.OBS_F_MIN, RE.OBS_F_MAX)

ORDERS_DEFAULT = (1, 2, 4, 8, 16, 32)
WINDOWS_DEFAULT = (0.128, 0.256, 0.512, 1.024, 2.048, 4.096)
NFFTS_DEFAULT = (2048, 4096, 16384)

#: Half-power width and +-2-bin share are read on the baseline-subtracted
#: shape; the baseline is the median PSD of the outer half of the baseband.
OUTER_FRACTION = 0.5

#: Second, NARROW band limit of the envelope measurement. The comb's own
#: isolation limit B is 15-38 Hz here, so an envelope read through it is
#: resolution-limited at 13-32 ms and cannot show a slow amplitude drift at
#: all. +-2 Hz resolves 0.25 s and slower, which is the scale a Welch window
#: has to stay inside.
ENVELOPE_NARROW_HZ = 2.0

# ── the frozen supports ─────────────────────────────────────────────────────

#: The five DREGON room-2 cruise scoring windows of
#: docs/revised-phase-baseline-manifest-v2.json (all mics, 4 s).
DREGON_SCORED: tuple[tuple[str, float, float], ...] = (
    ("free-flight_nosource_room2", 1512727397.2050455, 4.0),
    ("hovering_nosource_room2", 1511903905.3944898, 4.0),
    ("updown_nosource_room2", 1511903578.348311, 4.0),
    ("rectangle_nosource_room2", 1511905725.952559, 4.0),
    ("spinning_nosource_room2", 1511905200.978012, 4.0),
)

#: Michael's held-out FLY124 windows the frozen evaluator resolved (read from
#: the C3 eval_v2 metrics, so they are not re-derived here).
MICHAELS_SCORED: tuple[tuple[str, float, float, str], ...] = (
    ("FLY124", 8.0, 8.0, "standby"),
    ("FLY124", 16.0, 8.0, "standby"),
    ("FLY124", 27.68, 8.0, "ramp"),
    ("FLY124", 40.0, 8.0, "cruise"),
    ("FLY124", 56.0, 8.0, "cruise"),
)

#: Every TRAINING support that must stay out of a held-out measurement: the
#: legacy stage-2 exports' own 16 s windows AND the C3 fit windows. Both
#: families are excluded from every oracle and line-shape support, so no
#: number here is read on material any arm was fitted on.
TRAINING_SUPPORTS: dict[str, tuple[tuple[str, float, float], ...]] = {
    "dregon": (
        ("free-flight_nosource_room2", 1512727417.2050455, 16.0),
        ("hovering_nosource_room2", 1511903913.3944898, 16.0),
        ("updown_nosource_room2", 1511903588.177404, 16.0),
        ("rectangle_nosource_room2", 1511905733.5589993, 16.0),
        ("spinning_nosource_room2", 1511905207.3349512, 16.0),
    ),
    # Michael's fits are all on FLY125; FLY124 carries no training material.
    "michaels": (
        ("FLY125", 2.0, 8.0),
        ("FLY125", 10.0, 8.0),
        ("FLY125", 16.0, 16.0),
        ("FLY125", 32.0, 16.0),
        ("FLY125", 48.0, 16.0),
        ("FLY125", 64.0, 16.0),
        ("FLY125", 96.0, 16.0),
        ("FLY125", 112.0, 16.0),
        ("FLY125", 128.0, 16.0),
        ("FLY125", 144.0, 16.0),
    ),
}

REGIME_BANDS: dict[str, tuple[float, float | None]] = {
    "standby": (20.0, 45.0),
    "ramp": (45.0, 65.0),
    "cruise": (65.0, None),
}

DATASET = {"dregon": "DREGON-frames", "michaels": "michaels-frames"}
#: The carriers each rig supplies. DREGON publishes a raw command track and a
#: refined posterior label; Michael's publishes raw ``rps``.
CARRIERS = {"dregon": ("motors_command", "rps_refined"), "michaels": ("rps",)}
#: The RAW reference every scored arm is measured against (never the refined
#: posterior label).
RAW_KEY = {"dregon": "motors_command", "michaels": "rps"}

#: The current-best synthetic arm: the legacy stage-2 export the frozen
#: evaluator selected per cohort/regime, with the manifest's declared
#: provenance where the file carries no ``data`` block.
BASELINE_EXPORTS: dict[tuple[str, str], dict[str, Any]] = {
    ("dregon", "cruise"): dict(
        path="results/S2/dregon_room2_cruise_refined.json",
        family="refined",
        regime="cruise",
        declared=None,
        match="identity",
    ),
    ("michaels", "cruise"): dict(
        path="results/S2/cruise_8clip_refined.json",
        family="refined",
        regime="cruise",
        declared=None,
        match="aggregate",
    ),
    ("michaels", "ramp"): dict(
        path="results/S2/cruise_8clip_refined.json",
        family="refined",
        regime="cruise",
        declared=None,
        match="aggregate",
    ),
    ("michaels", "standby"): dict(
        path="results/S2/standby.json",
        family="raw",
        regime="standby",
        declared=dict(
            recordings=["FLY125"],
            starts_s=[1.78],
            seconds=13.5,
            dataset="michaels-frames",
            version=None,
            rps_key="rps",
        ),
        match="aggregate",
    ),
}

C3_EXPORTS = {
    "dregon": Path(
        ".worktrees/revised-phase-preflight/omnirun-outputs/"
        "revised-phase-c3-fits-ad2c57/results/revised_phase/c3/dregon.json"
    ),
    "michaels": Path(
        ".worktrees/revised-phase-preflight/omnirun-outputs/"
        "revised-phase-c3-fits-ad2c57/results/revised_phase/c3/michaels.json"
    ),
}


# ── provenance ──────────────────────────────────────────────────────────────


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover - provenance never aborts a run
        return f"unavailable: {exc}"


# ── supports ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Support:
    """One measured span, with the rule that produced it."""

    rig: str
    recording: str
    start_s: float
    duration_s: float
    regime: str
    role: str
    rule: str

    @property
    def window(self) -> RE.Window:
        return RE.Window(self.recording, self.start_s, self.duration_s, self.regime, self.role)

    def as_dict(self) -> dict[str, Any]:
        return dict(
            rig=self.rig,
            recording=self.recording,
            start_s=float(self.start_s),
            duration_s=float(self.duration_s),
            regime=self.regime,
            role=self.role,
            rule=self.rule,
            key=self.window.key,
        )


def guarded_training(rig: str, recording: str) -> list[tuple[float, float]]:
    """``[(lo, hi)]`` spans forbidden to a held-out measurement of ``recording``."""
    out = []
    for rec, start, seconds in TRAINING_SUPPORTS[rig]:
        if rec != recording:
            continue
        out.append((float(start) - GUARD_S, float(start) + float(seconds) + GUARD_S))
    return out


def _subtract(
    spans: list[tuple[float, float]], holes: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Set difference of interval lists, both on the recording's own clock."""
    out = list(spans)
    for lo, hi in holes:
        nxt: list[tuple[float, float]] = []
        for a, b in out:
            if hi <= a or lo >= b:
                nxt.append((a, b))
                continue
            if a < lo:
                nxt.append((a, min(lo, b)))
            if b > hi:
                nxt.append((max(hi, a), b))
        out = [(a, b) for a, b in nxt if b > a]
    return out


def legal_spans(
    rec: Any, rig: str, regime: str, *, apply_regime: bool
) -> list[tuple[float, float]]:
    """In-regime spans of one recording minus every guarded training support."""
    if apply_regime:
        lo, hi = REGIME_BANDS[regime]
        spans = RE.regime_intervals(rec, min_rps=lo, max_rps=hi)
    else:
        spans = [tuple(float(v) for v in rec.coverage)]  # type: ignore[misc]
    return _subtract(
        [(float(a), float(b)) for a, b in spans], guarded_training(rig, rec.recording_id)
    )


def lineshape_support(
    rec: Any, rig: str, scored: Support, *, cap_seconds: float, in_regime: bool
) -> Support:
    """The scored window extended FORWARD as far as the data legally allows.

    The rule, applied identically to every window: start at the frozen scoring
    start, extend while the span stays (a) inside coverage, (b) inside the same
    in-regime span when the frozen window is itself fully in regime, (c) clear
    of every guarded training support, and (d) under ``cap_seconds``. The
    scored window is therefore always a prefix of the line-shape support.
    """
    spans = legal_spans(rec, rig, scored.regime, apply_regime=in_regime)
    holder = [
        (a, b)
        for a, b in spans
        if a <= scored.start_s + 1e-6 and b >= scored.start_s + scored.duration_s - 1e-6
    ]
    if not holder:
        # the frozen window is not inside one legal in-regime span: keep it as
        # declared and say so rather than silently moving it
        return Support(
            rig,
            scored.recording,
            scored.start_s,
            scored.duration_s,
            scored.regime,
            "lineshape",
            "frozen scoring window kept verbatim: no single legal in-regime span contains it",
        )
    a, b = holder[0]
    dur = min(float(cap_seconds), b - scored.start_s)
    return Support(
        rig,
        scored.recording,
        scored.start_s,
        float(dur),
        scored.regime,
        "lineshape",
        f"frozen scoring start extended forward inside one legal span to {dur:.3f} s "
        f"(cap {cap_seconds:g} s, guard {GUARD_S} s)",
    )


def _mean_rps_lookup(rec: Any, *, step_s: float = 0.01) -> Any:
    """A cheap ``(t0, t1) -> per-rotor mean rps`` on one recording's clock."""
    lo, hi = (float(v) for v in rec.coverage)
    times = np.arange(lo, hi, float(step_s))
    block = np.asarray(rec.rps_at(times), dtype=np.float64)  # (R, N)
    cum = np.concatenate([np.zeros((block.shape[0], 1)), np.cumsum(block, axis=1)], axis=1)

    def mean_rps(t0: float, t1: float) -> np.ndarray:
        i = int(np.clip(round((t0 - lo) / step_s), 0, times.size - 1))
        j = int(np.clip(round((t1 - lo) / step_s), i + 1, times.size))
        return (cum[:, j] - cum[:, i]) / float(j - i)

    return mean_rps


def disjoint_segment(
    rec: Any,
    rig: str,
    scored: Support,
    *,
    min_seconds: float,
    in_regime: bool,
    step_s: float = 0.25,
) -> tuple[Support, dict[str, Any]] | None:
    """The oracle's real reference: a DISJOINT legal segment of the SAME recording.

    The oracle must be a FLOOR, so the segment is not simply the earliest
    legal one: an arbitrary later segment of a moving flight sits at other
    rotor speeds and other levels, and then real-vs-real reads WORSE than
    real-vs-synthetic (the synthetic arms are rendered on the scored window's
    own carrier). The selection rule is therefore stated and fixed:

    1. keep only spans that are legal (in regime, clear of every guarded
       training support) and disjoint from the scored window widened by the
       guard;
    2. take the longest achievable duration, capped at the scored duration --
       never padded;
    3. among every candidate start on a ``step_s`` grid at that duration,
       take the one whose per-rotor mean speed is closest to the scored
       window's (mean absolute difference over rotors), earliest on a tie.

    The achieved speed mismatch travels with the number, so a reader can see
    how well matched the floor is.
    """
    spans = legal_spans(rec, rig, scored.regime, apply_regime=in_regime)
    hole = [(scored.start_s - GUARD_S, scored.start_s + scored.duration_s + GUARD_S)]
    free = [(a, b) for a, b in _subtract(spans, hole) if b - a >= float(min_seconds)]
    if not free:
        return None
    dur = min(float(scored.duration_s), max(b - a for a, b in free))
    mean_rps = _mean_rps_lookup(rec)
    target = mean_rps(scored.start_s, scored.start_s + scored.duration_s)
    best: tuple[float, float] | None = None
    for a, b in free:
        if b - a < dur - 1e-9:
            continue
        n_steps = int(np.floor((b - a - dur) / float(step_s))) + 1
        for i in range(max(n_steps, 1)):
            t0 = a + i * float(step_s)
            mismatch = float(np.abs(mean_rps(t0, t0 + dur) - target).mean())
            if best is None or mismatch < best[0] - 1e-12:
                best = (mismatch, t0)
    assert best is not None
    mismatch, start = best
    sup = Support(
        rig,
        scored.recording,
        float(start),
        float(dur),
        scored.regime,
        "oracle_reference",
        f"legal disjoint span (guard {GUARD_S} s), {dur:.3f} s, start chosen on a {step_s:g} s grid "
        f"to minimise the per-rotor mean-speed mismatch ({mismatch:.3f} rev/s)",
    )
    return sup, dict(
        mean_rps_mismatch_rps=float(mismatch),
        scored_mean_rps=[float(v) for v in target],
        reference_mean_rps=[float(v) for v in mean_rps(start, start + dur)],
        n_candidate_starts=sum(
            max(int(np.floor((b - a - dur) / float(step_s))) + 1, 0)
            for a, b in free
            if b - a >= dur - 1e-9
        ),
        duration_s=float(dur),
    )


def load(support: Support, *, rps_key: str, channels: tuple[int, ...] | None) -> Clip:
    return RE.load_window(
        support.window,
        dataset=DATASET[support.rig],
        version=None,
        channels=channels,
        rps_key=rps_key,
        target_sr=SR,
    )


# ── demodulation and the baseband line shape ────────────────────────────────


def demodulate(audio: np.ndarray, rps_r: np.ndarray, k: int, *, sr: int = SR) -> np.ndarray:
    """``(M, T)`` complex baseband of harmonic ``k`` of one rotor.

    The carrier is the EXACT integrated telemetry phase
    ``2 pi k int_0^t f_r(tau) d tau`` (trapezoid on the audio grid), which is
    the deterministic term of the revised model. What survives the product is
    the modelled residual ``A exp{i[k theta_r + eps_rk + alpha]}`` plus
    whatever else the recording carries at those frequencies.
    """
    r = np.asarray(rps_r, dtype=np.float64)
    integral = (np.cumsum(r) - 0.5 * (r + r[0])) / float(sr)
    phase = 2.0 * np.pi * float(k) * integral
    return np.asarray(audio, dtype=np.float64) * np.exp(-1j * phase)[None, :]


def two_sided_welch(
    z: np.ndarray, seconds: float, *, sr: int = SR
) -> tuple[np.ndarray, np.ndarray] | None:
    """Two-sided Welch PSD of a complex signal: periodic Hann, 50 % overlap."""
    nper = int(round(float(seconds) * sr))
    if nper < 8 or nper > z.shape[-1]:
        return None
    window = np.hanning(nper + 1)[:nper]  # periodic Hann, the project's convention
    # the scipy stubs declare `window: str` and `detrend: str`; the array window
    # and `detrend=False` are the documented runtime forms, passed as kwargs
    kw: dict[str, Any] = dict(
        window=window,
        nperseg=nper,
        noverlap=nper // 2,
        detrend=False,
        scaling="density",
    )
    freqs, psd = signal.welch(z, fs=float(sr), return_onesided=False, **kw)
    order = np.argsort(freqs)
    return np.asarray(freqs)[order], np.asarray(psd)[..., order]


def n_welch_segments(n_samples: int, seconds: float, *, sr: int = SR) -> int:
    nper = int(round(float(seconds) * sr))
    if nper > n_samples:
        return 0
    return 1 + (n_samples - nper) // max(nper // 2, 1)


@dataclass
class LineShape:
    f: np.ndarray
    psd: np.ndarray
    baseline: float
    shape: np.ndarray  # baseline-subtracted, clipped at zero
    df: float
    b_hz: float
    snr_db: float
    half_power_width_hz: float | None
    width_censored: bool
    frac_within_2bins: float | None
    frac_within_2bins_raw: float
    frac_within_2bins_unclipped: float | None


def line_shape(f: np.ndarray, psd: np.ndarray, *, b_hz: float) -> LineShape:
    """Baseband line metrics inside the comb's own isolation limit ``b_hz``."""
    df = float(f[1] - f[0])
    keep = np.abs(f) <= b_hz + 1e-9
    fb, pb = f[keep], psd[keep]
    outer = np.abs(fb) >= OUTER_FRACTION * b_hz
    baseline = float(np.median(pb[outer])) if int(outer.sum()) >= 3 else 0.0
    near = np.abs(fb) <= 2.0 * df + 1e-9
    i0 = int(np.argmax(np.where(near, pb, -np.inf)))
    peak = float(pb[i0])
    snr_db = float(10.0 * np.log10(max(peak, 1e-300) / max(baseline, 1e-300)))
    shape = np.maximum(pb - baseline, 0.0)
    total = float(shape.sum())
    half = 0.5 * float(shape[i0])
    width: float | None = None
    censored = True
    if half > 0.0:
        left = right = None
        for j in range(i0, 0, -1):
            if shape[j] <= half:
                span = shape[j + 1] - shape[j]
                frac = 0.0 if span == 0.0 else (half - shape[j]) / span
                left = fb[j] + frac * (fb[j + 1] - fb[j])
                break
        for j in range(i0, shape.size - 1):
            if shape[j] <= half:
                span = shape[j - 1] - shape[j]
                frac = 0.0 if span == 0.0 else (half - shape[j]) / span
                right = fb[j] - frac * (fb[j] - fb[j - 1])
                break
        if left is not None and right is not None:
            width = float(right - left)
            censored = False
    return LineShape(
        f=fb,
        psd=pb,
        baseline=baseline,
        shape=shape,
        df=df,
        b_hz=float(b_hz),
        snr_db=snr_db,
        half_power_width_hz=width,
        width_censored=censored,
        frac_within_2bins=(None if total <= 0.0 else float(shape[near].sum() / total)),
        frac_within_2bins_raw=float(pb[near].sum() / max(float(pb.sum()), 1e-300)),
        frac_within_2bins_unclipped=(
            None
            if abs(float((pb - baseline).sum())) <= 0.0
            else float((pb - baseline)[near].sum() / (pb - baseline).sum())
        ),
    )


def carrier_drift(
    rps: np.ndarray, orders: list[int], *, sr: int = SR, smooth_s: float = 0.1
) -> dict[str, Any]:
    """How fast the telemetry carrier moves, and the window that survives it.

    A Welch window of length ``T`` resolves ``1.44 / T`` Hz (the periodic
    Hann's own half-power width). Over that window, harmonic ``k`` of a rotor
    drifting at ``|df_r/dt|`` rev/s per s moves ``k |df_r/dt| T`` Hz. The
    window stops being a line measurement when the drift exceeds the
    resolution, i.e. above

        T_max(k) = sqrt(1.44 / (k |df_r/dt|)).

    This is the non-stationarity limit that actually binds the window, and it
    is a property of the telemetry, not of the audio.
    """
    r = np.atleast_2d(np.asarray(rps, dtype=np.float64))
    w = max(int(round(float(smooth_s) * sr)), 3)
    kernel = np.ones(w) / float(w)
    slopes = []
    for row in r:
        smooth = np.convolve(row, kernel, mode="valid")
        slopes.append(np.abs(np.diff(smooth)) * float(sr))
    arr = np.concatenate(slopes) if slopes else np.zeros(1)
    med = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    return dict(
        abs_slope_rps_per_s=dict(median=med, p95=p95, max=float(arr.max())),
        smooth_s=float(smooth_s),
        t_max_s_at_median=[
            dict(order=int(k), t_max_s=(None if med <= 0 else float(np.sqrt(1.44 / (k * med)))))
            for k in orders
        ],
        t_max_s_at_p95=[
            dict(order=int(k), t_max_s=(None if p95 <= 0 else float(np.sqrt(1.44 / (k * p95)))))
            for k in orders
        ],
        rule="T_max(k) = sqrt(1.44 / (k * |d rps/dt|)): the window at which the line's drift "
        "equals the periodic Hann half-power width",
    )


def rebin_to(f_fine: np.ndarray, s_fine: np.ndarray, f_coarse: np.ndarray) -> np.ndarray:
    """Sum a fine shape into the coarse grid's bins (nearest-centre assignment)."""
    df = float(f_coarse[1] - f_coarse[0])
    idx = np.round((f_fine - f_coarse[0]) / df).astype(int)
    ok = (idx >= 0) & (idx < f_coarse.size)
    out = np.zeros(f_coarse.size, dtype=np.float64)
    np.add.at(out, idx[ok], s_fine[ok])
    return out


def shape_l1(coarse: LineShape, fine: LineShape) -> float | None:
    """L1 between the normalised line shapes at ``T`` and ``2T``, on ``T``'s grid."""
    a = coarse.shape
    b = rebin_to(fine.f, fine.shape, coarse.f)
    if a.sum() <= 0.0 or b.sum() <= 0.0:
        return None
    return float(np.abs(a / a.sum() - b / b.sum()).sum())


def shape_l1_same_grid(a: LineShape, b: LineShape) -> float | None:
    """L1 between two normalised shapes measured at the SAME window.

    The split-half NULL of :func:`shape_l1`: two disjoint halves of the same
    material, the same ``T``, the same grid. A Welch shape from a handful of
    segments carries real estimator noise, and without this null the
    ``T`` vs ``2T`` distance cannot be read as a resolution effect.
    """
    if a.f.size != b.f.size or a.shape.sum() <= 0.0 or b.shape.sum() <= 0.0:
        return None
    return float(np.abs(a.shape / a.shape.sum() - b.shape / b.shape.sum()).sum())


def line_envelope(z: np.ndarray, *, b_hz: float, sr: int = SR) -> tuple[np.ndarray, float]:
    """``(|A(t)|, sample rate)`` of the band-limited complex line amplitude.

    An exact FFT-domain band limit to ``+-b_hz`` followed by the matching
    decimation: the envelope cannot be resolved faster than ``1 / (2 b_hz)``,
    which is the comb's own isolation limit and is reported with the number.
    """
    n = int(z.shape[-1])
    spec = np.fft.fft(z, axis=-1)
    nb = int(np.floor(float(b_hz) * n / float(sr)))
    if nb < 4:
        return np.abs(z), float(sr)
    kept = np.concatenate([spec[..., : nb + 1], spec[..., n - nb :]], axis=-1)
    m = 2 * nb + 1
    baseband = np.fft.ifft(kept, axis=-1) * (float(m) / float(n))
    return np.abs(baseband), float(m) / (float(n) / float(sr))


def one_over_e_time(env: np.ndarray, sr_bb: float) -> float | None:
    """1/e time of the amplitude-envelope autocorrelation, linearly interpolated."""
    x = np.asarray(env, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    if n < 16 or not np.any(x):
        return None
    ac = np.correlate(x, x, mode="full")[n - 1 :]
    ac = ac / ac[0]
    target = float(np.exp(-1.0))
    below = np.flatnonzero(ac <= target)
    if below.size == 0:
        return None
    j = int(below[0])
    if j == 0:
        return 0.0
    hi, lo = ac[j - 1], ac[j]
    frac = 0.0 if hi == lo else (hi - target) / (hi - lo)
    return float((j - 1 + frac) / sr_bb)


# ── stage 1: the line-shape grid ────────────────────────────────────────────


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return dict(n=0, median=None, p25=None, p75=None, mean=None, min=None, max=None)
    return dict(
        n=int(arr.size),
        median=float(np.median(arr)),
        p25=float(np.percentile(arr, 25)),
        p75=float(np.percentile(arr, 75)),
        mean=float(arr.mean()),
        min=float(arr.min()),
        max=float(arr.max()),
    )


def lineshape_stage(
    *,
    rigs: list[str],
    orders: list[int],
    windows: list[float],
    n_mics: int,
    cap_seconds: float,
    supports_per_rig: int | None,
) -> dict[str, Any]:
    mics = tuple(range(int(n_mics)))
    payload: dict[str, Any] = dict(
        front_end=dict(
            sr=SR,
            window="periodic Hann",
            overlap="50 %",
            psd="two-sided Welch density of the complex demodulated baseband",
            carrier="2 pi k * trapezoid integral of the supplied rotor track",
            baseband_limit="B = 0.5 * min_r mean rps of the support (half the harmonic spacing)",
            baseline="median PSD over 0.5 B <= |f| <= B, subtracted before every shape metric",
            mics=list(mics),
            orders=list(orders),
            windows_s=list(windows),
        ),
        supports=[],
        cells=[],
        aggregate=[],
        envelope=[],
        shapes=[],
    )
    per_cell: dict[tuple[str, str, int, float], dict[str, list[float]]] = {}
    shape_bank: dict[tuple[str, str, int, float], list[LineShape]] = {}
    env_bank: dict[tuple[str, str, int, str], list[float]] = {}
    res_bank: dict[tuple[str, str, int, str], float] = {}

    for rig in rigs:
        scored_list: list[Support] = []
        if rig == "dregon":
            for rec, start, dur in DREGON_SCORED:
                scored_list.append(
                    Support(rig, rec, start, dur, "cruise", "scored", "frozen manifest v2")
                )
        else:
            for rec, start, dur, regime in MICHAELS_SCORED:
                scored_list.append(
                    Support(rig, rec, start, dur, regime, "scored", "frozen evaluator resolution")
                )
        if supports_per_rig is not None:
            scored_list = scored_list[: int(supports_per_rig)]
        for scored in scored_list:
            rec_obj = C.load_recording(DATASET[rig], scored.recording, None, RAW_KEY[rig])
            in_regime = scored.regime != "ramp"
            sup = lineshape_support(
                rec_obj, rig, scored, cap_seconds=cap_seconds, in_regime=in_regime
            )
            row = sup.as_dict()
            row["scored_window"] = scored.window.key
            row["in_regime_rule"] = in_regime
            payload["supports"].append(row)
            for label in CARRIERS[rig]:
                try:
                    clip = load(sup, rps_key=label, channels=mics)
                except Exception as exc:
                    row.setdefault("load_errors", {})[label] = f"{type(exc).__name__}: {exc}"
                    continue
                rps = np.atleast_2d(np.asarray(clip.rps, dtype=np.float64))
                audio = np.asarray(clip.audio, dtype=np.float64)
                rates = rps.mean(axis=1)
                b_hz = 0.5 * float(rates.min())
                row.setdefault("mean_rps", {})[label] = [float(v) for v in rates]
                row.setdefault("baseband_b_hz", {})[label] = b_hz
                lo, hi = REGIME_BANDS[scored.regime]
                keep = np.ones(rps.shape[1], dtype=bool)
                keep &= np.all(rps >= lo, axis=0)
                if hi is not None:
                    keep &= np.all(rps <= hi, axis=0)
                row.setdefault("regime_fraction", {})[label] = float(keep.mean())
                row.setdefault("carrier_drift", {})[label] = carrier_drift(rps, orders)
                for k in orders:
                    centres = float(k) * rates
                    if centres.max() > BAND_HZ[1] or centres.min() < BAND_HZ[0]:
                        continue
                    for r in range(rps.shape[0]):
                        z = demodulate(audio, rps[r], k)
                        # the line amplitude's own envelope, at TWO band limits: the
                        # comb's isolation limit B (fast, and resolution-limited at
                        # 1 / 2B) and a narrow +-2 Hz band that can only carry the
                        # SLOW amplitude modulation
                        for tag, bw in (("", b_hz), ("_narrow", ENVELOPE_NARROW_HZ)):
                            env, sr_bb = line_envelope(z, b_hz=bw)
                            for mi in range(env.shape[0]):
                                tau = one_over_e_time(env[mi], sr_bb)
                                if tau is not None:
                                    env_bank.setdefault((rig, label, k, tag), []).append(tau)
                                res_bank[(rig, label, k, tag)] = 1.0 / (2.0 * float(bw))
                        shapes: dict[float, list[LineShape]] = {}
                        halves: dict[float, tuple[list[LineShape], list[LineShape]]] = {}
                        n_half = audio.shape[1] // 2
                        for t_win in sorted(set(windows) | {2.0 * w for w in windows}):
                            out = two_sided_welch(z, t_win)
                            if out is None:
                                continue
                            f, psd = out
                            shapes[t_win] = [
                                line_shape(f, psd[mi], b_hz=b_hz) for mi in range(psd.shape[0])
                            ]
                            lo_out = two_sided_welch(z[..., :n_half], t_win)
                            hi_out = two_sided_welch(z[..., n_half:], t_win)
                            if lo_out is not None and hi_out is not None:
                                halves[t_win] = (
                                    [
                                        line_shape(lo_out[0], lo_out[1][mi], b_hz=b_hz)
                                        for mi in range(psd.shape[0])
                                    ],
                                    [
                                        line_shape(hi_out[0], hi_out[1][mi], b_hz=b_hz)
                                        for mi in range(psd.shape[0])
                                    ],
                                )
                        for t_win in windows:
                            if t_win not in shapes:
                                continue
                            fine = shapes.get(2.0 * t_win)
                            split = halves.get(t_win)
                            for mi, ls in enumerate(shapes[t_win]):
                                key = (rig, label, k, t_win)
                                acc = per_cell.setdefault(
                                    key,
                                    dict(
                                        width=[],
                                        width_clean=[],
                                        censored=[],
                                        frac2=[],
                                        frac2_clean=[],
                                        frac2raw=[],
                                        frac2unc=[],
                                        l1=[],
                                        l1_clean=[],
                                        l1_null=[],
                                        snr=[],
                                        clean=[],
                                    ),
                                )
                                clean = ls.snr_db >= RP.GATE_MIN_LINE_SNR_DB
                                acc["snr"].append(ls.snr_db)
                                acc["clean"].append(1.0 if clean else 0.0)
                                acc["censored"].append(1.0 if ls.width_censored else 0.0)
                                if ls.half_power_width_hz is not None:
                                    acc["width"].append(ls.half_power_width_hz)
                                    if clean:
                                        acc["width_clean"].append(ls.half_power_width_hz)
                                if ls.frac_within_2bins is not None:
                                    acc["frac2"].append(ls.frac_within_2bins)
                                    if clean:
                                        acc["frac2_clean"].append(ls.frac_within_2bins)
                                acc["frac2raw"].append(ls.frac_within_2bins_raw)
                                if ls.frac_within_2bins_unclipped is not None:
                                    acc["frac2unc"].append(ls.frac_within_2bins_unclipped)
                                l1 = None if fine is None else shape_l1(ls, fine[mi])
                                if l1 is not None:
                                    acc["l1"].append(l1)
                                    if clean:
                                        acc["l1_clean"].append(l1)
                                null = (
                                    None
                                    if split is None
                                    else shape_l1_same_grid(split[0][mi], split[1][mi])
                                )
                                if null is not None:
                                    acc["l1_null"].append(null)
                                if scored.regime == "cruise":
                                    # ONE regime per figure: B = f_r / 2 differs by a
                                    # factor of 2.4 between standby and cruise, and a
                                    # median over both would rebin two different bands
                                    # onto one grid. Every rotor and mic is banked.
                                    shape_bank.setdefault(key, []).append(ls)
                                payload["cells"].append(
                                    dict(
                                        rig=rig,
                                        label=label,
                                        support=sup.window.key,
                                        rotor=int(r),
                                        order=int(k),
                                        mic=int(mics[mi]),
                                        window_s=float(t_win),
                                        df_hz=ls.df,
                                        b_hz=ls.b_hz,
                                        n_segments=n_welch_segments(audio.shape[1], t_win),
                                        line_snr_db=ls.snr_db,
                                        clean=bool(clean),
                                        half_power_width_hz=ls.half_power_width_hz,
                                        width_censored=bool(ls.width_censored),
                                        frac_within_2bins=ls.frac_within_2bins,
                                        frac_within_2bins_raw=ls.frac_within_2bins_raw,
                                        frac_within_2bins_unclipped=ls.frac_within_2bins_unclipped,
                                        window_invariance_l1=l1,
                                        window_invariance_l1_null=null,
                                    )
                                )

    for (rig, label, k, t_win), acc in sorted(per_cell.items()):
        l1_med = _stats(acc["l1"])["median"]
        null_med = _stats(acc["l1_null"])["median"]
        payload["aggregate"].append(
            dict(
                rig=rig,
                label=label,
                order=int(k),
                window_s=float(t_win),
                df_hz=1.0 / float(t_win),
                half_power_width_hz=_stats(acc["width"]),
                half_power_width_hz_clean=_stats(acc["width_clean"]),
                width_censored_fraction=float(np.mean(acc["censored"]))
                if acc["censored"]
                else None,
                frac_within_2bins=_stats(acc["frac2"]),
                frac_within_2bins_clean=_stats(acc["frac2_clean"]),
                frac_within_2bins_raw=_stats(acc["frac2raw"]),
                frac_within_2bins_unclipped=_stats(acc["frac2unc"]),
                window_invariance_l1=_stats(acc["l1"]),
                window_invariance_l1_clean=_stats(acc["l1_clean"]),
                window_invariance_l1_null=_stats(acc["l1_null"]),
                window_invariance_excess=(
                    None if (l1_med is None or null_med is None) else float(l1_med - null_med)
                ),
                line_snr_db=_stats(acc["snr"]),
                clean_fraction=float(np.mean(acc["clean"])) if acc["clean"] else None,
                clean_rule=f"line SNR over the local baseband floor >= {RP.GATE_MIN_LINE_SNR_DB} dB "
                "(the preregistered moment-gate threshold)",
            )
        )
    for (rig, label, k, tag), taus in sorted(env_bank.items()):
        payload["envelope"].append(
            dict(
                rig=rig,
                label=label,
                order=int(k),
                band=("isolation_B" if tag == "" else f"narrow_{ENVELOPE_NARROW_HZ:g}Hz"),
                envelope_1e_time_s=_stats(taus),
                resolution_limit_s=res_bank.get((rig, label, k, tag)),
                note="amplitude-envelope autocorrelation 1/e time of the band-limited complex "
                "line; a median at the resolution limit means the envelope is not resolved "
                "by that band, not that it is that fast",
            )
        )
    for (rig, label, k, t_win), bank in sorted(shape_bank.items()):
        # median normalised shape over every banked (support, rotor, mic) of the
        # CRUISE supports, rebinned onto the widest banked grid
        grid = max(bank, key=lambda ls: ls.f.size)
        stack = [rebin_to(ls.f, ls.shape, grid.f) for ls in bank if ls.shape.sum() > 0.0]
        if not stack:
            continue
        norm = np.median(np.stack([s / s.sum() for s in stack]), axis=0)
        payload["shapes"].append(
            dict(
                rig=rig,
                label=label,
                order=int(k),
                window_s=float(t_win),
                f_hz=[float(v) for v in grid.f],
                normalised_shape=[float(v) for v in norm],
                b_hz=grid.b_hz,
                n_supports=len(stack),
            )
        )
    return payload


# ── stage 2: the composite risk of three arms ───────────────────────────────


def baseline_params(rig: str, regime: str, recording: str) -> RE.ModelParams:
    """The current-best arm's parameter set, on the evaluator's own route."""
    cfg = BASELINE_EXPORTS[(rig, regime)]
    bundle = RE.read_export(
        cfg["path"], family=cfg["family"], regime=cfg["regime"], declared=cfg["declared"]
    )
    if cfg["match"] == "identity":
        pairs = bundle.clips_of(recording)
        if not pairs:
            raise SystemExit(f"{cfg['path']}: no identity entry for {recording!r}")
        ids = [cid for cid, _ in pairs]
        extrapolated = False
    else:
        ids = list(bundle.clip_ids)
        extrapolated = True
    mp = RE.aggregate_nuisance(
        bundle,
        ids,
        label=f"{cfg['family']}:{cfg['regime']}:{cfg['match']}",
        extrapolated=extrapolated,
    )
    mp.source.update(for_recording=recording, baseline_route=cfg["match"], export=str(cfg["path"]))
    return mp


def oracle_mean_periodogram(clip: Clip, *, n_fft: int, hop: int, n_mics: int) -> np.ndarray:
    """``(M, 1, F)`` Welch MEAN PERIODOGRAM of the disjoint real segment.

    Exactly ``data.periodogram``'s geometry and normalisation, averaged over
    its frames: the non-parametric predictor of the scored window's mean
    spectrum, with no model and no carrier in it.
    """
    pg = periodogram(clip, n_fft=n_fft, hop=hop)
    return np.asarray(pg.power, dtype=np.float64)[:n_mics].mean(axis=1, keepdims=True)


def composite_stage(
    *, rigs: list[str], nffts: list[int], supports_per_rig: int | None, n_mics: int
) -> dict[str, Any]:
    payload: dict[str, Any] = dict(
        band_hz=list(BAND_HZ),
        arms=dict(
            oracle_np="Welch mean periodogram of a disjoint real segment of the same recording",
            current_best=(
                "legacy stage-2 export through revised_eval.predicted_m "
                f"({RE.HISTORICAL_FORWARD_LABEL})"
            ),
            c3="revised_phase.predict_spectrum(mode='prior') of the round-3 C3 export",
        ),
        supports=[],
        rows=[],
        summary=[],
    )
    items: dict[tuple[str, str, str, int], list[RE.FrameScore]] = {}
    for rig in rigs:
        c3 = RE.read_candidate_export(C3_EXPORTS[rig])
        payload.setdefault("c3_parameters", {})[rig] = {
            k: float(c3.summary["parameters"][k]) for k in ("lam", "sigma", "d_scalar")
        }
        scored_list: list[Support] = []
        if rig == "dregon":
            for rec, start, dur in DREGON_SCORED:
                scored_list.append(
                    Support(rig, rec, start, dur, "cruise", "scored", "frozen manifest v2")
                )
        else:
            for rec, start, dur, regime in MICHAELS_SCORED:
                scored_list.append(
                    Support(rig, rec, start, dur, regime, "scored", "frozen evaluator resolution")
                )
        if supports_per_rig is not None:
            scored_list = scored_list[: int(supports_per_rig)]
        for scored in scored_list:
            rec_obj = C.load_recording(DATASET[rig], scored.recording, None, RAW_KEY[rig])
            in_regime = scored.regime != "ramp"
            found = disjoint_segment(rec_obj, rig, scored, min_seconds=2.048, in_regime=in_regime)
            ref, ref_detail = (None, None) if found is None else found
            clip = load(scored, rps_key=RAW_KEY[rig], channels=None)
            mics = min(int(n_mics), int(clip.audio.shape[0]))
            ref_clip = None if ref is None else load(ref, rps_key=RAW_KEY[rig], channels=None)
            entry = dict(
                scored=scored.as_dict(),
                oracle_reference=None if ref is None else ref.as_dict(),
                oracle_match=ref_detail,
            )
            payload["supports"].append(entry)
            mp = baseline_params(rig, scored.regime, scored.recording)
            entry["baseline_source"] = dict(mp.source)
            for n_fft in nffts:
                hop = n_fft // 16
                pg = periodogram(clip, n_fft=n_fft, hop=hop)
                band = RE.observation_band(pg.freqs)
                power = np.asarray(pg.power, dtype=np.float64)[:mics]
                models: dict[str, np.ndarray | None] = {}
                t0 = time.time()
                models["current_best"] = RE.predicted_m(mp, pg, n_mics=mics)
                t_best = time.time() - t0
                t0 = time.time()
                m_c3 = np.asarray(
                    RP.predict_spectrum(c3.summary, clip, n_fft=n_fft, hop=hop, mode="prior"),
                    dtype=np.float64,
                )
                models["c3"] = m_c3[:mics]
                t_c3 = time.time() - t0
                if ref_clip is not None:
                    models["oracle_np"] = np.broadcast_to(
                        oracle_mean_periodogram(ref_clip, n_fft=n_fft, hop=hop, n_mics=mics),
                        power.shape,
                    )
                else:
                    models["oracle_np"] = None
                for arm, model in models.items():
                    if model is None:
                        payload["rows"].append(
                            dict(
                                rig=rig,
                                regime=scored.regime,
                                support=scored.window.key,
                                arm=arm,
                                n_fft=int(n_fft),
                                hop=int(hop),
                                available=False,
                                reason="no legal disjoint real segment of this recording",
                            )
                        )
                        continue
                    nll, cells = RE.marginal_frame_nll(power, model, band)
                    fs = RE.FrameScore(
                        scored.window,
                        RE.frame_times_on_clock(scored.window, pg),
                        nll,
                        cells,
                        int(n_fft),
                        int(hop),
                    )
                    items.setdefault((rig, scored.regime, arm, n_fft), []).append(fs)
                    one = RE.composite_score([fs])
                    payload["rows"].append(
                        dict(
                            rig=rig,
                            regime=scored.regime,
                            support=scored.window.key,
                            arm=arm,
                            n_fft=int(n_fft),
                            hop=int(hop),
                            available=True,
                            n_mics=mics,
                            n_frames=int(one["n_frames"]),
                            band_bins=int(band.sum()),
                            score=float(one["score"]),
                            score_per_band_cell=float(one["score_per_band_cell"]),
                            unique_seconds=float(one["unique_seconds"]),
                            model_seconds=(
                                t_best if arm == "current_best" else (t_c3 if arm == "c3" else 0.0)
                            ),
                        )
                    )
    for (rig, regime, arm, n_fft), fs_list in sorted(items.items()):
        comp = RE.composite_score(fs_list)
        per = [
            float(r["score"])
            for r in payload["rows"]
            if r.get("available")
            and (r["rig"], r["regime"], r["arm"], r["n_fft"]) == (rig, regime, arm, n_fft)
        ]
        payload["summary"].append(
            dict(
                rig=rig,
                regime=regime,
                arm=arm,
                n_fft=int(n_fft),
                hop=int(n_fft) // 16,
                pooled_score=float(comp["score"]),
                pooled_score_per_band_cell=float(comp["score_per_band_cell"]),
                unique_seconds=float(comp["unique_seconds"]),
                n_supports=len(fs_list),
                per_support_mean=float(np.mean(per)) if per else None,
                per_support_std=float(np.std(per, ddof=1)) if len(per) > 1 else None,
            )
        )
    return payload


# ── gap closure ─────────────────────────────────────────────────────────────


def gap_closure(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``(bad - x) / (bad - oracle)`` per (rig, regime, NFFT): the HPPNet form.

    1.0 is the oracle, 0.0 is the known-bad C3 arm. Stated on the pooled
    score AND on the per-band-cell score, because the two differ only by a
    constant and Main may prefer either.
    """
    keyed = {(r["rig"], r["regime"], r["arm"], r["n_fft"]): r for r in summary}
    out = []
    for (rig, regime, arm, n_fft), row in sorted(keyed.items()):
        oracle = keyed.get((rig, regime, "oracle_np", n_fft))
        bad = keyed.get((rig, regime, "c3", n_fft))
        if oracle is None or bad is None:
            continue
        span = bad["pooled_score"] - oracle["pooled_score"]
        out.append(
            dict(
                rig=rig,
                regime=regime,
                arm=arm,
                n_fft=int(n_fft),
                score=row["pooled_score"],
                oracle=oracle["pooled_score"],
                c3=bad["pooled_score"],
                span=float(span),
                closure=None
                if span == 0.0
                else float((bad["pooled_score"] - row["pooled_score"]) / span),
                score_per_band_cell=row["pooled_score_per_band_cell"],
                oracle_per_band_cell=oracle["pooled_score_per_band_cell"],
                c3_per_band_cell=bad["pooled_score_per_band_cell"],
            )
        )
    return out


# ── figures ─────────────────────────────────────────────────────────────────


def figure_lineshape(payload: dict[str, Any], fig_dir: Path, out_dir: Path) -> list[str]:
    names = []
    shapes = payload.get("shapes", [])
    keys = sorted({(s["rig"], s["label"]) for s in shapes})
    for rig, label in keys:
        rows = [s for s in shapes if s["rig"] == rig and s["label"] == label]
        orders = sorted({s["order"] for s in rows})
        if not orders:
            continue
        ncol = min(3, len(orders))
        nrow = int(np.ceil(len(orders) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.3 * nrow), squeeze=False)
        wins = sorted({s["window_s"] for s in rows})
        cmap = plt.get_cmap("viridis")
        agg = payload.get("aggregate", [])
        for ax, k in zip(axes.ravel(), orders, strict=False):
            b_hz = None
            for wi, t_win in enumerate(wins):
                sel = [s for s in rows if s["order"] == k and s["window_s"] == t_win]
                if not sel:
                    continue
                s = sel[0]
                b_hz = s["b_hz"]
                f = np.asarray(s["f_hz"])
                p = np.asarray(s["normalised_shape"])
                dens = p / max(float(f[1] - f[0]), 1e-12)
                # a log axis cannot show a clipped zero; floor four decades under
                # the peak so the visible dynamic range is the same in every panel
                floor = max(float(dens.max()) * 1e-4, 1e-12)
                ax.semilogy(
                    f,
                    np.maximum(dens, floor),
                    color=cmap(wi / max(len(wins) - 1, 1)),
                    lw=1.2,
                    label=f"T={t_win:g} s",
                )
            snr = next(
                (
                    a["line_snr_db"]["median"]
                    for a in agg
                    if (a["rig"], a["label"], a["order"]) == (rig, label, k)
                    and a["window_s"] == max(wins)
                ),
                None,
            )
            title = f"k = {k}"
            if snr is not None:
                title += f"   (median line SNR {snr:.1f} dB at T={max(wins):g} s)"
            ax.set_title(title, fontsize=9)
            if b_hz is not None:
                ax.set_xlim(-b_hz, b_hz)
            ax.set_xlabel("baseband offset (Hz)", fontsize=8)
            ax.set_ylabel("normalised density (1/Hz)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
        for ax in axes.ravel()[len(orders) :]:
            ax.axis("off")
        axes.ravel()[0].legend(fontsize=6, ncol=2)
        fig.suptitle(
            f"{rig} / carrier {label}: demodulated line shape against the Welch window\n"
            "median over every held-out CRUISE support, rotor and mic; baseline-subtracted, "
            "unit area inside the comb's isolation limit |f| <= B = f_r / 2",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        name = f"criteria_lineshape_{rig}_{label}.png"
        for d in (fig_dir, out_dir):
            fig.savefig(d / name, dpi=140)
        plt.close(fig)
        names.append(name)
    agg = payload.get("aggregate", [])
    if agg:
        combos = sorted({(a["rig"], a["label"]) for a in agg})
        fig, axes = plt.subplots(4, len(combos), figsize=(4.6 * len(combos), 11.4), squeeze=False)
        for ci, (rig, label) in enumerate(combos):
            rows = [a for a in agg if a["rig"] == rig and a["label"] == label]
            orders = sorted({a["order"] for a in rows})
            cmap = plt.get_cmap("plasma")
            for oi, k in enumerate(orders):
                sel = sorted([a for a in rows if a["order"] == k], key=lambda a: a["window_s"])
                t = [a["window_s"] for a in sel]
                col = cmap(oi / max(len(orders) - 1, 1))
                axes[0][ci].plot(
                    t,
                    [a["half_power_width_hz"]["median"] for a in sel],
                    "o-",
                    color=col,
                    lw=1.1,
                    ms=3,
                    label=f"k={k}",
                )
                axes[1][ci].plot(
                    t,
                    [a["frac_within_2bins"]["median"] for a in sel],
                    "o-",
                    color=col,
                    lw=1.1,
                    ms=3,
                )
                axes[2][ci].plot(
                    t,
                    [a["window_invariance_l1"]["median"] for a in sel],
                    "o-",
                    color=col,
                    lw=1.1,
                    ms=3,
                )
                axes[2][ci].plot(
                    t,
                    [a["window_invariance_l1_null"]["median"] for a in sel],
                    ":",
                    color=col,
                    lw=1.0,
                )
                axes[3][ci].plot(
                    t, [a["line_snr_db"]["median"] for a in sel], "o-", color=col, lw=1.1, ms=3
                )
            t_ref = np.asarray(sorted({a["window_s"] for a in rows}), dtype=np.float64)
            axes[0][ci].plot(t_ref, 1.44 / t_ref, "k--", lw=1.4, label="1.44/T (Hann resolution)")
            axes[3][ci].axhline(
                RP.GATE_MIN_LINE_SNR_DB, color="k", ls="--", lw=1.4, label="10 dB gate"
            )
            for row in range(4):
                ax = axes[row][ci]
                ax.set_xscale("log")
                ax.grid(alpha=0.25)
                ax.set_xlabel("Welch window T (s)", fontsize=8)
                ax.tick_params(labelsize=7)
            axes[0][ci].set_yscale("log")
            axes[0][ci].set_title(f"{rig} / {label}", fontsize=9)
            axes[0][ci].set_ylabel("half-power width (Hz)", fontsize=8)
            axes[1][ci].set_ylabel("power share within +-2 bins", fontsize=8)
            axes[2][ci].set_ylabel("L1(T, 2T)   dotted: split-half null", fontsize=8)
            axes[3][ci].set_ylabel("median line SNR (dB)", fontsize=8)
            axes[0][ci].legend(fontsize=6, ncol=2)
            axes[3][ci].legend(fontsize=6)
        fig.suptitle(
            "window statistics of the demodulated line (median over supports, rotors, mics)\n"
            "a measured width ON the 1.44/T line means the line is UNRESOLVED at that window",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.955))
        name = "criteria_lineshape_summary.png"
        for d in (fig_dir, out_dir):
            fig.savefig(d / name, dpi=140)
        plt.close(fig)
        names.append(name)
    return names


def figure_composite(payload: dict[str, Any], fig_dir: Path, out_dir: Path) -> list[str]:
    summary = payload.get("summary", [])
    if not summary:
        return []
    combos = sorted({(s["rig"], s["regime"]) for s in summary})
    fig, axes = plt.subplots(1, len(combos), figsize=(4.0 * len(combos), 3.8), squeeze=False)
    style = {
        "oracle_np": ("C2", "o", "ORACLE-NP (real vs disjoint real)"),
        "current_best": ("C0", "s", "current-best synthetic"),
        "c3": ("C3", "^", "C3 (known bad)"),
    }
    for ci, (rig, regime) in enumerate(combos):
        ax = axes[0][ci]
        for arm, (col, mk, lab) in style.items():
            sel = sorted(
                [s for s in summary if (s["rig"], s["regime"], s["arm"]) == (rig, regime, arm)],
                key=lambda s: s["n_fft"],
            )
            if not sel:
                continue
            ax.plot(
                [s["n_fft"] for s in sel],
                [s["pooled_score_per_band_cell"] for s in sel],
                marker=mk,
                color=col,
                lw=1.3,
                ms=5,
                label=lab,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([2048, 4096, 16384])
        ax.set_xticklabels(["2048", "4096", "16384"])
        ax.set_xlabel("NFFT (hop = NFFT/16)", fontsize=8)
        ax.set_ylabel("composite risk per band cell (nats)", fontsize=8)
        ax.set_title(f"{rig} / {regime}", fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=6)
    fig.suptitle(
        "frozen composite risk of three arms against analysis resolution (lower is better)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    name = "criteria_composite_vs_nfft.png"
    for d in (fig_dir, out_dir):
        fig.savefig(d / name, dpi=140)
    plt.close(fig)
    return [name]


# ── main ────────────────────────────────────────────────────────────────────


def parse_list(text: str, cast: Any) -> list[Any]:
    return [cast(v) for v in str(text).replace(",", " ").split()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--fig-dir", type=Path, default=FIG_DEFAULT)
    ap.add_argument(
        "--stage",
        choices=("all", "lineshape", "composite", "figures"),
        default="all",
        help="'figures' recomputes nothing and redraws from the existing likelihood.json",
    )
    ap.add_argument("--rigs", default="dregon,michaels")
    ap.add_argument("--orders", default=",".join(str(k) for k in ORDERS_DEFAULT))
    ap.add_argument("--windows", default=",".join(f"{w:g}" for w in WINDOWS_DEFAULT))
    ap.add_argument("--nffts", default=",".join(str(n) for n in NFFTS_DEFAULT))
    ap.add_argument("--lineshape-mics", type=int, default=2)
    ap.add_argument("--composite-mics", type=int, default=8)
    ap.add_argument("--cap-seconds", type=float, default=16.0)
    ap.add_argument("--supports-per-rig", type=int, default=None)
    ap.add_argument("--threads", type=int, default=0, help="torch threads; 0 leaves the default")
    args = ap.parse_args()

    if args.threads > 0:
        import torch

        torch.set_num_threads(int(args.threads))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rigs = parse_list(args.rigs, str)
    orders = parse_list(args.orders, int)
    windows = parse_list(args.windows, float)
    nffts = parse_list(args.nffts, int)

    t_start = time.time()
    payload: dict[str, Any] = dict(
        schema="noise-v2-criteria-likelihood/1",
        provenance=dict(
            git_head=git_head(),
            import_provenance=RE.import_provenance(),
            argv=vars(args) | dict(out=str(args.out), fig_dir=str(args.fig_dir)),
            guard_seconds=GUARD_S,
            training_supports={k: [list(v) for v in vs] for k, vs in TRAINING_SUPPORTS.items()},
            baseline_exports={f"{k[0]}:{k[1]}": v for k, v in BASELINE_EXPORTS.items()},
            c3_exports={k: str(v) for k, v in C3_EXPORTS.items()},
        ),
    )
    figures: list[str] = []
    existing = out_dir / "likelihood.json"
    prior: dict[str, Any] = {}
    if existing.is_file():
        try:
            prior = json.loads(existing.read_text())
        except Exception:
            prior = {}

    if args.stage in ("all", "lineshape"):
        print("[lineshape] start", flush=True)
        payload["lineshape"] = lineshape_stage(
            rigs=rigs,
            orders=orders,
            windows=windows,
            n_mics=int(args.lineshape_mics),
            cap_seconds=float(args.cap_seconds),
            supports_per_rig=args.supports_per_rig,
        )
        print(f"[lineshape] {len(payload['lineshape']['cells'])} cells", flush=True)
        figures += figure_lineshape(payload["lineshape"], fig_dir, out_dir)
    elif prior.get("lineshape"):
        payload["lineshape"] = prior["lineshape"]
        figures += figure_lineshape(payload["lineshape"], fig_dir, out_dir)

    if args.stage in ("all", "composite"):
        print("[composite] start", flush=True)
        payload["composite"] = composite_stage(
            rigs=rigs,
            nffts=nffts,
            supports_per_rig=args.supports_per_rig,
            n_mics=int(args.composite_mics),
        )
        payload["composite"]["gap_closure"] = gap_closure(payload["composite"]["summary"])
        figures += figure_composite(payload["composite"], fig_dir, out_dir)
    elif prior.get("composite"):
        payload["composite"] = prior["composite"]
        figures += figure_composite(payload["composite"], fig_dir, out_dir)

    payload["figures"] = figures
    payload["provenance"]["wall_seconds"] = float(time.time() - t_start)
    RE.write_json(out_dir / "likelihood.json", payload)
    print(
        f"wrote {out_dir / 'likelihood.json'} ({(out_dir / 'likelihood.json').stat().st_size / 1e6:.2f} MB)"
    )
    print("figures:", ", ".join(figures))
    print(f"total wall {payload['provenance']['wall_seconds']:.1f} s")


if __name__ == "__main__":
    main()
