"""Baseline instrumentation for the revised rotor-noise phase.

Everything the gate runner (``scripts/stochastic_fit_revised_eval.py``) needs
that is not a CLI: window identity, the support guard, export provenance and
identity pairing, nuisance aggregation, the declared render-unit conversion,
the legacy predicted-``M`` adapter, the composite spectral risk and the gate
arithmetic with its clustered intervals.

Six contracts this module exists to hold, all of them frozen upstream:

* **Identity pairing.** A baseline export is matched to a recording by
  ``data.recordings`` (or by a manifest-declared provenance for a legacy
  export that predates that block). There is no positional or modulo
  fallback: an unmatched recording is an error, never clip ``i % n``.
* **One timebase convention: the recording's own clock.** DREGON window
  starts are absolute (``1512727417.205``), Michael's are relative
  (``16.0``), and both are exactly what :meth:`clips.Recording.cut` takes,
  because ``Recording.t_start`` carries the offset. Nothing here assumes a
  zero origin; spans are compared against ``Recording.coverage``, never
  against 0.
* **Report, never pad.** A recording that cannot supply the requested
  disjoint held-out support is reported as insufficient, and an insufficient
  support fails the gate rather than shrinking the cohort.
* **Duplicate-ANALYSIS-FRAME exposure invariance.** Exposure is carried by
  the analysis frames, not by the clip spans: an observation is identified by
  ``(recording, absolute frame centre, n_fft)``, identical copies split one
  unit of exposure between them, and every frame carries ``hop / n_fft`` of
  it. Duplicating a single frame entry — not merely a whole clip — leaves the
  risk unchanged, and so does re-running the same material at a different hop
  density.
* **Exact timelines.** An arm is scored on the frozen sample length and the
  frozen microphone set; a short or narrow render is REJECTED, never
  truncated onto a different timeline.
* **Declared units.** The render path's absolute level is converted by a
  stated factor (:func:`to_renderer_units`), validated against a planted tone
  and a planted floor. No silent factor anywhere.

The composite score is a positive exposure-weighted marginal ``I / M + log M``
normalized per unique observed support. At the frozen convention the analysis
frames overlap **16-fold** (``N_FFT = 16384``, ``HOP = 1024``), so it is a
COMPOSITE RISK — proper for mean prediction — and NOT an exact joint NLL, not
evidence and not a posterior; a fixed hop is a decision-risk setting, never a
statement of information content. It is never reported as a percentage.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from experiments.stochastic_fit import accept_stats as stats
from experiments.stochastic_fit import clips as C
from experiments.stochastic_fit import phase_kernel as PK
from experiments.stochastic_fit.data import Clip, Periodogram, periodogram
from experiments.stochastic_fit.stage2 import SR as STAGE2_SR

SCHEMA = "revised-phase-baseline/1"

#: The FROZEN spectral front end, READ FROM THE SHARED KERNEL
#: (:mod:`experiments.stochastic_fit.phase_kernel`, owned by the candidate
#: worker) so the two adapters cannot drift: 16 kHz clips, periodic Hann of
#: 16384 points, a 1024-sample (64 ms) hop and a fixed 30-7900 Hz band on a
#: fixed microphone set. The band is never lowered to hide filter loss.
OBS_N_FFT, OBS_HOP, SR = PK.N_FFT, PK.HOP, PK.SR
OBS_F_MIN, OBS_F_MAX = PK.BAND_HZ
if int(STAGE2_SR) != int(SR):
    raise RuntimeError(f"stage2.SR {STAGE2_SR} disagrees with phase_kernel.SR {SR}")

#: What ``predicted_m`` is, stated wherever it is consumed (addendum 4 §23,
#: option 1). The candidate's own spectrum comes from
#: ``revised_phase.predict_spectrum``, which is the exact moving-window law.
HISTORICAL_FORWARD_LABEL = (
    "historical-forward baseline diagnostic: the LEGACY descriptive model's own forward "
    "prediction (model.CombSpectrum — frame-mean carrier plus a known chirp-width covariate), "
    "NOT the exact moving-window kernel the candidate path uses. Its ramp prediction in "
    "particular is an approximation the contract rejects for a candidate."
)

#: Travels with every saved number (frozen addendum 3 section 21).
ADAPTIVE_SELECTION_CAVEAT = (
    "Benchmark selection is adaptive over up to 8 candidate rounds, so these bootstrap and t "
    "intervals are decision/uncertainty summaries, not fresh-unseen confirmation. This record "
    "is not an independent final generalization proof."
)

#: Rotor tracks that are RAW telemetry, i.e. admissible as the primary scoring
#: reference. ``rps_refined`` is a posterior track: it is fit provenance, never
#: the reference a synthetic arm is scored against (frozen protocol section 7).
RAW_RPS_KEYS = ("rps", "motors_measured", "motors_command", "motors_command_raw")
REFINED_RPS_KEY = "rps_refined"

#: dB fields of an export that are POWER quantities: aggregated in linear
#: power, never in dB. The first two are absolute levels and therefore carry
#: the fit's per-clip ``power_scale``; the rest are relative patterns.
ABSOLUTE_POWER_DB_FIELDS = ("profile_db", "floor_mean_db")
RELATIVE_POWER_DB_FIELDS = ("floor_shape_db", "mic_floor_db", "mic_gain_db", "gain_all_db")
#: Fields averaged arithmetically: rates, exponents, shares, dB-per-octave.
LINEAR_FIELDS = (
    "gamma0",
    "gamma_slope",
    "width_power",
    "coherence_k_half",
    "floor_tilt_db_oct",
    "amp_exp",
    "floor_exp",
    "floor_static_rel",
)
#: Clip-local latents: a fitted draw for ONE window, not a rig property. They
#: are dropped when an export is used to predict another window, and the drop
#: is recorded in the aggregate's provenance.
CLIP_LOCAL_LATENTS = ("h_db", "floor_level_db", "floor_tilt_gp", "rps_offset", "umod_db", "carrier")


# ── the mandatory import-provenance guard ───────────────────────────────────

#: The DECLARED grids (addendum 11 §55). The word "native" is retired: there is
#: a MODEL/RENDER grid (``sample_rate_work``, an integer 4x oversampling of the
#: analysis rate — 64000, never 65536) and an ANALYSIS grid (16 kHz, where
#: ``data.periodogram`` reads). ``clips.decimate`` and all real-data sampling
#: are untouched by either.
SAMPLE_RATE_WORK = 64000
ANALYSIS_GRID_HZ = SR


def import_provenance(*, expected_root: str | Path | None = None) -> dict[str, Any]:
    """Where the modules under test ACTUALLY came from, as a checkable record.

    The shared ``.venv`` carries an editable install pointing at the MAIN
    checkout's ``src``, and ``stage2.py`` exists in both checkouts, so a run
    whose ``sys.path`` does not put this worktree first can exercise main's
    copy while still appearing to pass. Modules that exist only here would fail
    loudly; ``stage2`` would not. So every validation run and every smoke proof
    records and ASSERTS the resolved files of ``experiments``,
    ``revised_phase``, ``stage2`` (plus this module and the shared kernel)
    against one checkout root — this module's own — and against
    ``expected_root`` when the caller states one (the runner passes the cwd).

    Raises :class:`RuntimeError` on any mismatch. A run whose import
    provenance is unproven is not validation.
    """
    import experiments
    from experiments.stochastic_fit import phase_kernel, revised_phase, stage2

    here = Path(__file__).resolve()
    root = here.parents[3]  # <checkout>/src/experiments/stochastic_fit/<this file>
    modules = {
        "experiments": experiments,
        "revised_phase": revised_phase,
        "stage2": stage2,
        "phase_kernel": phase_kernel,
        "revised_eval": sys.modules[__name__],
    }
    resolved = {
        name: str(Path(str(getattr(mod, "__file__", ""))).resolve())
        for name, mod in modules.items()
    }
    root_str = str(root)
    outside = {n: p for n, p in resolved.items() if not p.startswith(root_str + "/")}
    record: dict[str, Any] = dict(
        checkout_root=root_str,
        cwd=str(Path.cwd().resolve()),
        expected_root=(str(Path(expected_root).resolve()) if expected_root else None),
        modules=resolved,
        sys_path_head=[str(p) for p in sys.path[:4]],
        rule=(
            "every module must resolve inside ONE checkout — this module's own — because the "
            "shared .venv's editable install also exposes the main checkout's src"
        ),
    )
    if outside:
        raise RuntimeError(
            f"import provenance failed: {outside} resolve outside the checkout {root_str}. "
            "Put <checkout>/src first on PYTHONPATH (or use the worktree's direnv env); "
            "never run uv sync."
        )
    if expected_root is not None:
        want = Path(expected_root).resolve()
        if root != want:
            raise RuntimeError(
                f"import provenance failed: modules resolve to {root_str} but this run expects "
                f"{want}. The shared .venv's editable install points at the main checkout."
            )
    record["verified"] = True
    return record


# ── the declared conversion from stored fit power to renderer units ─────────


def rate_factor_db(sample_rate_work: int = SAMPLE_RATE_WORK, sample_rate: int = SR) -> float:
    """``10 log10(work / analysis)`` — the sample-rate term of the conversion."""
    return float(10.0 * math.log10(float(sample_rate_work) / float(sample_rate)))


#: +6.0206 dB at the declared 64 kHz work grid; +4.4032 dB on S2's historical
#: 44.1 kHz path, which remains available as context only.
RENDER_RATE_FACTOR_DB = rate_factor_db()
HISTORICAL_RATE_FACTOR_DB = rate_factor_db(C.NATIVE_SR)


def to_renderer_units(
    mp: ModelParams,
    *,
    sample_rate_work: int = SAMPLE_RATE_WORK,
    sample_rate: int = SR,
) -> ModelParams:
    """A legacy export in PHYSICAL renderer units, by the declared conversion.

    Two separate effects, named separately (frozen addendum 7 §38, addendum 8
    §43); this function owns the second one, and the first is switched off by
    the caller:

    1. ``normalize_rms`` — ``stochastic_rotor_noise.synthesize`` rescales the
       WHOLE waveform to a fixed RMS by default (0.1), which erases the
       absolute level while leaving line-to-floor ratios intact. The evaluator
       passes ``normalize_rms=None`` explicitly; every other caller keeps the
       old default. Measured on a planted clip: the default moved both the
       lines and the floor by a common +11.1 dB, i.e. a pure gain, entirely
       determined by that clip's own RMS.
    2. **The sample-rate term, ``10 log10(work / analysis)``** — +6.0206 dB on
       the declared 64 kHz work grid (+4.4032 dB on the historical 44.1 kHz
       path). A periodogram in ``data.periodogram`` units reads ``PSD * sr/2``,
       so a floor SHAPED on the work grid reads ``sr_analysis / sr_work`` lower
       once decimated. In PHYSICAL terms this term belongs to the floor and not
       to a tone (decimation preserves a tone's amplitude), but the LEGACY line
       parameter is not an amplitude: ``model.CombSpectrum.line_power`` is a
       band weight in periodogram*Hz (addendum 7 §38 conversion 1), i.e. an
       integral that carries the same ``sr / 2``. So in THIS parameterization
       the term applies to ``profile_db`` and ``floor_mean_db`` alike. The
       candidate export's ``profile_db`` is a mean square (``A = sqrt(2P)``)
       and must NEVER be given this term — see
       :func:`legacy_profile_db_to_candidate` for the one stated factor to use
       if a level is ever seeded across families.

    Validated by planting known levels and measuring the rendered, decimated
    clip on the frozen observation grid (1 rotor, 40 orders, flat floor, no
    speed law, all-coherent lines, ``normalize_rms=None``) at BOTH synthesis
    rates; the numbers live in the proposals doc and in
    ``tests/experiments/test_stochastic_fit_render_timebase.py``.

    LEGACY EXPORTED LEVELS ARE STALE as absolute numbers: every published
    render they were checked against went through ``normalize_rms=0.1``. Only
    an export that has had its ``power_scale`` folded in
    (:func:`aggregate_nuisance`) and this term applied is a physical level, and
    even then it is the source-side quantity of ONE declared observation chain
    (:data:`stage2.RENDER_TRANSFER_SPEC`), never a calibrated absolute source
    spectrum.
    """
    off = rate_factor_db(sample_rate_work, sample_rate)
    params = dict(mp.params)
    for key in ABSOLUTE_POWER_DB_FIELDS:
        if key in params:
            params[key] = (np.asarray(params[key], dtype=np.float64) + off).tolist()
    source = dict(mp.source)
    source["render_units"] = dict(
        rate_factor_db=off,
        sample_rate_work=int(sample_rate_work),
        analysis_grid_hz=int(sample_rate),
        applied_to=list(ABSOLUTE_POWER_DB_FIELDS),
        normalize_rms="None (the evaluator's physical-level path)",
        derivation=(
            "I = |FFT(x w)|^2 / sum(w^2) reads PSD * sr/2, so a floor shaped on the work grid "
            "reads sr_analysis/sr_work lower after decimation; the legacy line parameter is an "
            "integral in periodogram*Hz and carries the same factor"
        ),
        validation="pure tone and floor planted at both synthesis rates (see to_renderer_units)",
    )
    return ModelParams(params=params, spec=mp.spec, source=source)


def legacy_profile_db_to_candidate(profile_db: np.ndarray | float) -> np.ndarray:
    """``P_candidate = 2 * P_legacy_physical / 16000`` in dB (addendum 9 §47).

    The two families' profile units differ by this fixed factor: the legacy
    number is a band weight in periodogram*Hz, the candidate's is a tone mean
    square (``A = sqrt(2P)``). Provided so that IF a level is ever seeded or
    compared across families the factor is applied explicitly and visibly; the
    evaluator itself never copies ``profile_db`` between families, and the two
    adapters stay on wholly separate declared units.
    """
    off = 10.0 * math.log10(2.0 / float(SR))
    return np.asarray(profile_db, dtype=np.float64) + off


def artifact_digest(path: str | Path) -> dict[str, Any]:
    """``{path, sha256, bytes}`` of a referenced artifact, for the frozen record.

    The frozen-manifest promise covers the artifacts the manifest POINTS AT,
    not only its own text: a baseline export that changes between ``--prepare``
    and ``--check`` would silently change the supports.
    """
    p = Path(path)
    raw = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


# ── window identity ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Window:
    """One concrete window: the identity everything is paired on.

    ``start_s`` is on the recording's own clock — absolute unix seconds for
    DREGON, relative seconds for Michael's — which is exactly what
    :meth:`clips.Recording.cut` and :func:`clips.load_clip` take.
    """

    recording: str
    start_s: float
    duration_s: float
    regime: str = "cruise"
    role: str = "evaluation"  # calibration | evaluation

    @property
    def end_s(self) -> float:
        return float(self.start_s) + float(self.duration_s)

    @property
    def key(self) -> str:
        """The pairing key: recording identity plus the exact span."""
        return f"{self.recording}@{self.start_s:.6f}+{self.duration_s:.6f}"

    @property
    def clip_id(self) -> str:
        return f"revised_{self.recording}_{self.start_s:.3f}+{self.duration_s:g}"

    def overlaps(self, other: Window, *, tol: float = 1e-9) -> bool:
        if self.recording != other.recording:
            return False
        return self.start_s < other.end_s - tol and other.start_s < self.end_s - tol

    def as_dict(self) -> dict[str, Any]:
        return dict(
            recording=self.recording,
            start_s=float(self.start_s),
            duration_s=float(self.duration_s),
            regime=self.regime,
            role=self.role,
            key=self.key,
        )


def union_seconds(windows: Sequence[Window]) -> float:
    """Seconds of UNIQUE material covered, per recording and summed.

    The denominator of the composite score: duplicated or overlapping windows
    do not inflate it.
    """
    total = 0.0
    by_rec: dict[str, list[tuple[float, float]]] = {}
    for w in windows:
        by_rec.setdefault(w.recording, []).append((float(w.start_s), w.end_s))
    for spans in by_rec.values():
        lo, hi = None, None
        for a, b in sorted(spans):
            if lo is None:
                lo, hi = a, b
                continue
            assert hi is not None
            if a <= hi:
                hi = max(hi, b)
            else:
                total += hi - lo
                lo, hi = a, b
        if lo is not None and hi is not None:
            total += hi - lo
    return float(total)


# ── the window-support guard ────────────────────────────────────────────────


@dataclass(frozen=True)
class SupportReport:
    """What a recording could and could not supply. Never a silent pad."""

    recording: str
    regime: str
    requested_seconds: float
    requested_windows: int
    n_in_regime: int
    n_disjoint: int
    kept: tuple[str, ...]
    dropped_overlapping: tuple[str, ...] = ()
    sufficient: bool = True
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return dict(
            recording=self.recording,
            regime=self.regime,
            requested_seconds=float(self.requested_seconds),
            requested_windows=int(self.requested_windows),
            n_in_regime=int(self.n_in_regime),
            n_disjoint=int(self.n_disjoint),
            disjoint_seconds=float(self.n_disjoint * self.requested_seconds),
            kept=list(self.kept),
            dropped_overlapping=list(self.dropped_overlapping),
            sufficient=bool(self.sufficient),
            note=self.note,
        )


def resolve_evaluation_windows(
    rec: Any,
    *,
    regime: str,
    calibration: Sequence[Window],
    min_seconds: float,
    stride_s: float | None = None,
    max_windows: int = 1,
    min_rps: float | None = None,
    max_rps: float | None = None,
) -> tuple[list[Window], SupportReport]:
    """Held-out windows of one recording, disjoint from its fit supports.

    The fit supports are training/calibration material, so a predictive number
    may only be read on what is left. A recording that cannot supply
    ``max_windows`` x ``min_seconds`` of disjoint in-regime material inside its
    label coverage is REPORTED as insufficient — the caller never receives a
    padded, repeated or cycled window.
    """
    found = C.windows(
        rec,
        seconds=float(min_seconds),
        max_clips=10_000,
        min_rps=min_rps,
        max_rps=max_rps,
        stride_s=float(stride_s or min_seconds),
    )
    cal = [w for w in calibration if w.recording == rec.recording_id]
    candidates = [
        Window(rec.recording_id, float(t0), float(dur), regime=regime, role="evaluation")
        for t0, dur in found
    ]
    dropped = [w.key for w in candidates if any(w.overlaps(c) for c in cal)]
    disjoint = [w for w in candidates if not any(w.overlaps(c) for c in cal)]
    kept = disjoint[: max(int(max_windows), 0)]
    note = ""
    if len(kept) < int(max_windows):
        note = (
            f"{rec.recording_id}: {len(kept)} of {max_windows} disjoint {regime} windows of "
            f"{min_seconds:g}s available (in-regime {len(candidates)}, "
            f"{len(dropped)} overlap the fit supports) — reported, not padded"
        )
    return kept, SupportReport(
        recording=rec.recording_id,
        regime=regime,
        requested_seconds=float(min_seconds),
        requested_windows=int(max_windows),
        n_in_regime=len(candidates),
        n_disjoint=len(disjoint),
        kept=tuple(w.key for w in kept),
        dropped_overlapping=tuple(dropped),
        sufficient=len(kept) >= int(max_windows),
        note=note,
    )


def regime_intervals(
    rec: Any,
    *,
    min_rps: float | None,
    max_rps: float | None,
    step_s: float = 0.01,
) -> list[tuple[float, float]]:
    """Contiguous spans of one regime on the recording's own clock.

    Read from the RAW telemetry at ``step_s`` resolution inside
    :attr:`clips.Recording.coverage`. These are the PRE-REGISTERED regime
    boundaries: they come from the manifest's band and the telemetry, never
    from a candidate and never from a refined track.
    """
    lo, hi = rec.coverage
    if hi - lo <= step_s:
        return []
    times = np.arange(float(lo), float(hi), float(step_s))
    block = rec.rps_at(times)
    keep = np.ones(times.shape, dtype=bool)
    if min_rps is not None:
        keep &= np.all(block >= float(min_rps), axis=0)
    if max_rps is not None:
        keep &= np.all(block <= float(max_rps), axis=0)
    padded = np.concatenate(([False], keep, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [
        (float(times[a]), float(times[b - 1] + step_s))
        for a, b in zip(edges[0::2], edges[1::2], strict=True)
    ]


def resolve_support_windows(
    rec: Any,
    *,
    regime: str,
    calibration: Sequence[Window],
    context_seconds: float,
    min_support_seconds: float,
    max_windows: int = 1,
    min_rps: float | None = None,
    max_rps: float | None = None,
) -> tuple[list[Window], SupportReport]:
    """CONTEXT windows around a regime whose support is SHORTER than a window.

    A transition is not 8 s long: FLY124's whole ramp band holds about 1.2 s of
    material, the longest contiguous run being under one second. Such a regime
    can never be found by asking for a window that is in-regime throughout, so
    the window here is CONTEXT — it carries the render, the analysis window and
    the filter guard — while the SCORED support inside it stays the
    pre-registered interval (:func:`regime_support` recomputes exactly that
    mask from the same band). The context may extend into other regimes; the
    scored material may not.

    A context window is kept only when it lies inside the label coverage and
    its scored interval is disjoint from every fit support. Everything that
    cannot be supplied is reported, never padded.
    """
    lo, hi = rec.coverage
    cal = [w for w in calibration if w.recording == rec.recording_id]
    spans = regime_intervals(rec, min_rps=min_rps, max_rps=max_rps)
    long_enough = [(a, b) for a, b in spans if (b - a) >= float(min_support_seconds)]
    kept: list[Window] = []
    dropped: list[str] = []
    for a, b in sorted(long_enough, key=lambda s: s[0] - s[1]):  # longest first
        centre = 0.5 * (a + b)
        start = min(
            max(centre - 0.5 * float(context_seconds), float(lo)),
            float(hi) - float(context_seconds),
        )
        if start < float(lo) - 1e-9:
            dropped.append(f"{a:.3f}+{b - a:.3f}:coverage")
            continue
        w = Window(
            rec.recording_id, float(start), float(context_seconds), regime=regime, role="evaluation"
        )
        support = Window(rec.recording_id, float(a), float(b - a), regime=regime)
        if any(support.overlaps(c) for c in cal):
            dropped.append(f"{support.key}:fit-support")
            continue
        if any(w.overlaps(k) for k in kept):
            continue  # one context window per piece of material
        kept.append(w)
        if len(kept) >= max(int(max_windows), 0):
            break
    note = ""
    if len(kept) < int(max_windows):
        longest = max((b - a for a, b in spans), default=0.0)
        note = (
            f"{rec.recording_id}: {len(kept)} of {max_windows} {regime} context windows of "
            f"{context_seconds:g}s available; the band holds {len(spans)} interval(s), "
            f"{sum(b - a for a, b in spans):.3f}s in total, longest {longest:.3f}s, and "
            f"{min_support_seconds:g}s of scored support was required — reported, not padded"
        )
    return kept, SupportReport(
        recording=rec.recording_id,
        regime=regime,
        requested_seconds=float(context_seconds),
        requested_windows=int(max_windows),
        n_in_regime=len(spans),
        n_disjoint=len(long_enough),
        kept=tuple(w.key for w in kept),
        dropped_overlapping=tuple(dropped),
        sufficient=len(kept) >= int(max_windows),
        note=note,
    )


def check_explicit_windows(
    windows: Sequence[Window],
    *,
    regime: str,
    calibration: Sequence[Window],
    requested: int | None = None,
) -> tuple[list[Window], list[SupportReport]]:
    """The same guard for manifest-declared windows: drop overlaps, report them."""
    reports: list[SupportReport] = []
    kept: list[Window] = []
    for rid in sorted({w.recording for w in windows}):
        mine = [w for w in windows if w.recording == rid]
        cal = [w for w in calibration if w.recording == rid]
        bad = [w for w in mine if any(w.overlaps(c) for c in cal)]
        good = [w for w in mine if not any(w.overlaps(c) for c in cal)]
        want = int(requested if requested is not None else len(mine))
        kept.extend(good)
        reports.append(
            SupportReport(
                recording=rid,
                regime=regime,
                requested_seconds=float(mine[0].duration_s),
                requested_windows=want,
                n_in_regime=len(mine),
                n_disjoint=len(good),
                kept=tuple(w.key for w in good),
                dropped_overlapping=tuple(w.key for w in bad),
                sufficient=len(good) >= want,
                note=(
                    ""
                    if len(good) >= want
                    else f"{rid}: {len(bad)} declared window(s) overlap the fit supports "
                    "and were dropped — reported, not replaced"
                ),
            )
        )
    return kept, reports


# ── exports: provenance and identity pairing ────────────────────────────────


@dataclass(frozen=True)
class ExportBundle:
    """A fit summary plus the provenance needed to pair it by identity.

    Newer exports carry a ``data`` block (``recordings``, ``clips``,
    ``starts_s``, ``seconds``, ``dataset``, ``rps_key``). Legacy exports
    predate it, so their provenance must be DECLARED in the manifest and is
    flagged as such everywhere it is reported.
    """

    path: Path
    family: str
    regime: str
    summary: dict[str, Any]
    recordings: tuple[str, ...]
    clip_ids: tuple[str, ...]
    starts_s: tuple[float, ...]
    seconds: float
    dataset: str | None
    version: str | None
    rps_key: str | None
    declared_provenance: bool

    @property
    def spec(self) -> dict[str, Any]:
        return dict(self.summary.get("spec") or {})

    def entry(self, clip_id: str) -> dict[str, Any]:
        return dict(self.summary["clips"][clip_id])

    def clips_of(self, recording: str) -> tuple[tuple[str, float], ...]:
        """``((clip_id, start_s), ...)`` of ONE recording, matched by identity."""
        out = [
            (cid, start)
            for cid, rid, start in zip(self.clip_ids, self._owner_ids(), self.starts_s, strict=True)
            if rid == recording
        ]
        return tuple(out)

    def _owner_ids(self) -> tuple[str, ...]:
        """Which recording each clip belongs to, by identity, never by index.

        A clip id produced by the fit is ``<recording>_<regime>_<nn>`` with the
        recording lower-cased, so the owner is recovered by prefix match
        against the export's own recording list. One recording (Michael's
        8-clip cruise fit) needs no matching at all.
        """
        if len(self.recordings) == 1:
            return tuple(self.recordings[0] for _ in self.clip_ids)
        owners: list[str] = []
        for cid in self.clip_ids:
            hit = [r for r in self.recordings if cid.lower().startswith(r.lower())]
            if len(hit) != 1:
                raise ValueError(
                    f"{self.path}: clip {cid!r} matches {len(hit)} of the export's recordings "
                    f"{list(self.recordings)} — identity pairing needs exactly one owner"
                )
            owners.append(hit[0])
        return tuple(owners)

    def calibration_windows(self) -> list[Window]:
        """The fit supports, as windows on each recording's own clock."""
        return [
            Window(rid, float(start), float(self.seconds), regime=self.regime, role="calibration")
            for rid, start in zip(self._owner_ids(), self.starts_s, strict=True)
        ]

    def provenance(self) -> dict[str, Any]:
        return dict(
            path=str(self.path),
            family=self.family,
            regime=self.regime,
            recordings=list(self.recordings),
            clips=list(self.clip_ids),
            starts_s=[float(s) for s in self.starts_s],
            seconds=float(self.seconds),
            dataset=self.dataset,
            dataset_version=self.version,
            fit_rps_key=self.rps_key,
            declared_provenance=bool(self.declared_provenance),
        )


def export_family(summary: Mapping[str, Any]) -> str:
    """``"legacy"`` or the export's own ``model_family`` — the dispatch test.

    A revised export carries ``model_family`` (``"shared_shaft_ou"``),
    ``parameters`` and ``training_provenance``, and has NO ``clips`` block. If
    ``model_family`` is absent the file is a legacy descriptive-model export.
    Evaluating a shape-compatible file through the wrong renderer or the wrong
    expected-periodogram law is a silent wrong answer, so the two paths never
    share code below this test.
    """
    family = summary.get("model_family")
    return "legacy" if family is None else str(family)


def read_export(
    path: str | Path,
    *,
    family: str,
    regime: str | None = None,
    declared: Mapping[str, Any] | None = None,
) -> ExportBundle:
    """Read a LEGACY fit summary and resolve its provenance.

    ``declared`` supplies the provenance of a legacy export (no ``data``
    block): ``recordings``, ``starts_s``, ``seconds`` and optionally
    ``dataset``/``version``/``rps_key``. A legacy export with no declaration
    is refused rather than paired positionally.

    A revised (``model_family``) export is REFUSED here: it must go through
    :func:`read_candidate_export` and the candidate-owned API.
    """
    p = Path(path)
    summary = json.loads(p.read_text())
    fam = export_family(summary)
    if fam != "legacy":
        raise ValueError(
            f"{p}: model_family={fam!r} is a revised export, not a legacy one — it must be "
            "evaluated through revised_phase.render_revised / predict_spectrum, never through "
            "the legacy renderer and the historical-forward adapter"
        )
    data = dict(summary.get("data") or {})
    clip_ids = tuple(str(c) for c in (data.get("clips") or list(summary["clips"].keys())))
    declared = dict(declared or {})
    if data.get("recordings"):
        recordings = tuple(str(r) for r in data["recordings"])
        starts = tuple(float(s) for s in data["starts_s"])
        seconds = float(data["seconds"])
        dataset, version = data.get("dataset"), data.get("dataset_version")
        rps_key = data.get("rps_key")
        declared_flag = False
    elif declared.get("recordings"):
        recordings = tuple(str(r) for r in declared["recordings"])
        starts = tuple(float(s) for s in declared["starts_s"])
        seconds = float(declared["seconds"])
        dataset, version = declared.get("dataset"), declared.get("version")
        rps_key = declared.get("rps_key")
        declared_flag = True
    else:
        raise ValueError(
            f"{p}: no `data.recordings` block and no declared provenance — a legacy export "
            "must have its recordings/starts_s/seconds declared in the manifest; pairing by "
            "position or modulo is not available"
        )
    if len(starts) != len(clip_ids):
        raise ValueError(
            f"{p}: {len(clip_ids)} clips but {len(starts)} starts — provenance does not describe "
            "this export"
        )
    return ExportBundle(
        path=p,
        family=str(family),
        regime=str(regime or data.get("regime") or summary.get("regime") or "cruise"),
        summary=summary,
        recordings=recordings,
        clip_ids=clip_ids,
        starts_s=starts,
        seconds=seconds,
        dataset=dataset,
        version=version,
        rps_key=rps_key,
        declared_provenance=declared_flag,
    )


# ── the revised candidate export: provenance, pinning, leakage ──────────────


@dataclass(frozen=True)
class CandidateExport:
    """A revised (``model_family``) export, read for provenance and dispatch.

    This module never implements the candidate model: it reads the layout the
    candidate's owner froze — ``schema_version``, ``model_family``, ``rig_id``,
    ``parameters``, ``training_provenance``, ``diagnostics`` — and dispatches
    to ``revised_phase.render_revised`` / ``revised_phase.predict_spectrum``.
    """

    path: Path
    summary: dict[str, Any]
    model_family: str
    schema_version: int
    rig_id: str
    digest: dict[str, Any]

    @property
    def n_rotors(self) -> int:
        return int(np.atleast_2d(np.asarray(self.summary["parameters"]["profile_db"])).shape[0])

    @property
    def n_orders(self) -> int:
        return int(np.atleast_2d(np.asarray(self.summary["parameters"]["profile_db"])).shape[1])

    @property
    def identified(self) -> bool:
        """The dynamics-identification flag from the export's diagnostics."""
        diag = dict(self.summary.get("diagnostics") or {})
        return bool(diag.get("identified", False))

    @property
    def provenance_block(self) -> dict[str, Any]:
        return dict(self.summary.get("training_provenance") or {})

    def training_windows(self) -> list[Window]:
        """The clips the candidate was FITTED on, as windows on their own clock."""
        out: list[Window] = []
        for entry in self.provenance_block.get("clips") or []:
            e = dict(entry)
            out.append(
                Window(
                    str(e["recording"]),
                    float(e["start_s"]),
                    float(e["seconds"]),
                    regime=str(e.get("regime", "unknown")),
                    role="candidate_training",
                )
            )
        return out

    def provenance(self) -> dict[str, Any]:
        prov = self.provenance_block
        return dict(
            path=str(self.path),
            digest=self.digest,
            model_family=self.model_family,
            schema_version=self.schema_version,
            rig_id=self.rig_id,
            n_rotors=self.n_rotors,
            n_orders=self.n_orders,
            fit_manifest_sha256=prov.get("manifest_sha256"),
            front_end=prov.get("front_end"),
            training_clips=[w.as_dict() for w in self.training_windows()],
            training_rps_keys=sorted(
                {str(dict(e).get("rps_key")) for e in (prov.get("clips") or [])}
            ),
        )


def read_candidate_export(path: str | Path) -> CandidateExport:
    """Read and pin a revised export; refuse a legacy one and refuse a stranger."""
    p = Path(path)
    summary = json.loads(p.read_text())
    fam = export_family(summary)
    if fam == "legacy":
        raise ValueError(
            f"{p}: no model_family — this is a legacy descriptive export. The candidate arm "
            "needs a revised export; a legacy file belongs to a baseline family."
        )
    for key in ("parameters", "training_provenance"):
        if not isinstance(summary.get(key), dict):
            raise ValueError(f"{p}: model_family={fam!r} but {key!r} is missing or not a mapping")
    identified = bool(dict(summary.get("diagnostics") or {}).get("identified", False))
    if fam == "shared_shaft_ou" and not identified:
        raise ValueError(
            f"{p}: model_family={fam!r} but diagnostics.identified is {identified!r}. "
            "An unidentifiable dynamics fit has no meaningful lam/sigma to compare; it must not "
            "be scored. If the field is missing, the export is malformed."
        )
    return CandidateExport(
        path=p,
        summary=summary,
        model_family=fam,
        schema_version=int(summary.get("schema_version", 0)),
        rig_id=str(summary.get("rig_id", "")),
        digest=artifact_digest(p),
    )


def training_leakage(
    training: Sequence[Window],
    scored: Sequence[Window],
    *,
    guard_seconds: float = 0.0,
) -> dict[str, Any]:
    """Overlap between a candidate's fit supports and the SCORED supports.

    The F5 "no held-out raw-sample support leakage" check at manifest level: a
    candidate that saw a scored window during fitting is not held out on it.
    ``guard_seconds`` widens each training span symmetrically, so an analysis
    or filter support that reaches into a scored window counts as overlap too.
    """
    hits: list[dict[str, Any]] = []
    for t in training:
        wide = Window(
            t.recording,
            float(t.start_s) - float(guard_seconds),
            float(t.duration_s) + 2.0 * float(guard_seconds),
            regime=t.regime,
            role=t.role,
        )
        for s in scored:
            if wide.overlaps(s):
                hits.append(
                    dict(
                        training_window=t.key,
                        training_regime=t.regime,
                        scored_window=s.key,
                        scored_regime=s.regime,
                        overlap_s=float(min(wide.end_s, s.end_s) - max(wide.start_s, s.start_s)),
                    )
                )
    return dict(
        clean=not hits,
        guard_seconds=float(guard_seconds),
        n_training_windows=len(training),
        n_scored_windows=len(scored),
        overlaps=hits,
        rule="a candidate fit support may never intersect a scored held-out support",
    )


# ── nuisance aggregation ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelParams:
    """Export parameters made level-explicit, i.e. in PHYSICAL power units.

    A fit scales its periodogram to unit mean (``scores.power_scale``) and
    exports the parameters in that scaled unit, so two clips' absolute levels
    are not comparable until the scale is folded back in. Aggregation and the
    predicted-``M`` adapter both work on physical units; the render path is
    untouched (it RMS-normalizes anyway).
    """

    params: dict[str, Any]
    spec: dict[str, Any]
    source: dict[str, Any]

    @property
    def n_orders(self) -> int:
        return int(np.asarray(self.params["profile_db"]).shape[1])


def _to_physical(entry: Mapping[str, Any]) -> dict[str, Any]:
    """One clip's export in physical units: fold in its own ``power_scale``."""
    params = {k: v for k, v in dict(entry["params"]).items()}
    scale = float(dict(entry.get("scores") or {}).get("power_scale", 1.0))
    off = 10.0 * math.log10(scale) if scale > 0 else 0.0
    for key in ABSOLUTE_POWER_DB_FIELDS:
        if key in params:
            params[key] = (np.asarray(params[key], dtype=np.float64) + off).tolist()
    params["power_scale"] = 1.0
    params["power_scale_folded_db"] = off
    return params


def _db_power_mean(values: Sequence[np.ndarray]) -> np.ndarray:
    """Mean of dB quantities in LINEAR POWER, back to dB."""
    lin = np.mean([10.0 ** (np.asarray(v, dtype=np.float64) / 10.0) for v in values], axis=0)
    return 10.0 * np.log10(np.maximum(lin, 1e-300))


def aggregate_nuisance(
    bundle: ExportBundle, clip_ids: Sequence[str], *, label: str, extrapolated: bool = False
) -> ModelParams:
    """One parameter set from several fitted clips of ONE rig.

    Levels and patterns are averaged in LINEAR POWER after each clip's own
    ``power_scale`` is folded in — never in dB, and never by taking whichever
    clip happens to be last. Rates, exponents and shares are averaged
    arithmetically. The order grid is truncated to the shortest clip's ladder,
    which is reported. Clip-local latents (line/floor drift draws, the carrier
    correction, the per-mic modulation) are DROPPED: they are one window's
    posterior draw, not a rig property.
    """
    if not clip_ids:
        raise ValueError(f"{bundle.path}: no clips selected for {label!r}")
    entries = [_to_physical(bundle.entry(cid)) for cid in clip_ids]
    k_use = min(int(np.asarray(e["profile_db"]).shape[1]) for e in entries)
    n_rotors = min(int(np.asarray(e["profile_db"]).shape[0]) for e in entries)
    out: dict[str, Any] = {}
    prof = [np.asarray(e["profile_db"], dtype=np.float64)[:n_rotors, :k_use] for e in entries]
    out["profile_db"] = _db_power_mean(prof).tolist()
    out["floor_mean_db"] = float(_db_power_mean([np.asarray(e["floor_mean_db"]) for e in entries]))
    for key in RELATIVE_POWER_DB_FIELDS:
        present = [np.asarray(e[key], dtype=np.float64) for e in entries if key in e]
        if len(present) == len(entries) and present:
            out[key] = _db_power_mean(present).tolist()
    for key in LINEAR_FIELDS:
        present = [np.asarray(e[key], dtype=np.float64) for e in entries if key in e]
        if len(present) == len(entries) and present:
            mean = np.mean(present, axis=0)
            out[key] = float(mean) if mean.ndim == 0 else mean.tolist()
    for key in ("floor_ctrl_hz",):
        if key in entries[0]:
            out[key] = np.asarray(entries[0][key], dtype=np.float64).tolist()
    out["power_scale"] = 1.0
    source = dict(
        label=label,
        export=str(bundle.path),
        family=bundle.family,
        regime=bundle.regime,
        clips=list(clip_ids),
        n_clips=len(clip_ids),
        recordings=sorted({r for r in bundle._owner_ids() if r}),
        aggregation="linear-power mean of per-clip nuisance, power_scale folded in per clip",
        power_scale_folded_db=[float(e["power_scale_folded_db"]) for e in entries],
        orders_used=int(k_use),
        rotors_used=int(n_rotors),
        dropped_clip_local_latents=[
            k for k in CLIP_LOCAL_LATENTS if k in bundle.entry(clip_ids[0])["params"]
        ],
        extrapolated=bool(extrapolated),
        declared_provenance=bool(bundle.declared_provenance),
    )
    return ModelParams(params=out, spec=bundle.spec, source=source)


# ── baseline family selection and the baseline map ──────────────────────────


@dataclass(frozen=True)
class FamilyCoverage:
    family: str
    regime: str
    path: str
    covered: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        return not self.missing

    def as_dict(self) -> dict[str, Any]:
        return dict(
            family=self.family,
            regime=self.regime,
            export=self.path,
            covered=list(self.covered),
            missing=list(self.missing),
            eligible=self.eligible,
            note=(
                ""
                if self.eligible
                else "ineligible: the export does not name every cohort recording, and pairing "
                "by modulo is not available"
            ),
        )


def family_coverage(
    bundles: Mapping[str, ExportBundle], recordings: Sequence[str]
) -> dict[str, FamilyCoverage]:
    """Which baseline families can supply an identity-matched entry per recording."""
    out: dict[str, FamilyCoverage] = {}
    for fam, b in bundles.items():
        covered = tuple(r for r in recordings if b.clips_of(r))
        out[fam] = FamilyCoverage(
            family=fam,
            regime=b.regime,
            path=str(b.path),
            covered=covered,
            missing=tuple(r for r in recordings if r not in covered),
        )
    return out


def select_family(
    scores: Mapping[str, float],
    coverage: Mapping[str, FamilyCoverage],
    *,
    require_coverage: bool = True,
) -> dict[str, Any]:
    """Freeze ONE baseline family for the whole rig/regime.

    ``scores`` are calibration-support scores, lower is better, and the choice
    is global — never per window. With ``require_coverage`` only a family that
    names every cohort recording by identity may win; that is the DREGON rule.
    Michael's held-out FLY124 has no export naming it at all, so there the
    baseline is a declared extrapolation and coverage cannot be required — the
    families are still compared on their own (FLY125) calibration supports.
    """
    eligible = {f: s for f, s in scores.items() if (coverage[f].eligible or not require_coverage)}
    chosen = min(eligible, key=lambda f: eligible[f]) if eligible else None
    return dict(
        chosen=chosen,
        scope="global per rig/regime — no per-window winners",
        criterion="calibration-support composite score (lower is better)",
        require_identity_coverage=bool(require_coverage),
        scores={f: float(s) for f, s in scores.items()},
        coverage={f: c.as_dict() for f, c in coverage.items()},
    )


# ── the predicted-M adapter ─────────────────────────────────────────────────


def _inv_softplus(x: float, *, floor: float = -30.0) -> float:
    v = float(x)
    if v <= 0.0:
        return floor
    return float(math.log(math.expm1(v))) if v < 30.0 else v


def _predict_block(
    mp: ModelParams,
    freqs: np.ndarray,
    times: np.ndarray,
    rps: np.ndarray,
    *,
    n_mics: int,
    band_top: float,
) -> np.ndarray:
    """``(M, n, F)`` expected periodogram for one block of frames."""
    import torch

    from experiments.stochastic_fit.model import CombSpectrum, Spec

    params = mp.params
    profile = np.atleast_2d(np.asarray(params["profile_db"], dtype=np.float64))
    live = rps[rps > 5.0]
    slowest = float(live.min()) if live.size else 20.0
    k_use = max(int(min(math.floor(band_top / max(slowest, 1e-3)), profile.shape[1])), 1)

    fields = {
        k: v
        for k, v in mp.spec.items()
        if k not in ("freqs", "times", "rps", "n_mics", "n_harm", "extra")
    }
    # the drift draws belong to the fitted window, not to the rig
    fields.update(gp_std_db=0.0, floor_gp_std_db=0.0, floor_tilt_gp_std=0.0, umod_std_db=0.0)
    fields["f_max"] = band_top
    spec = Spec(
        freqs=np.asarray(freqs, dtype=np.float64),
        times=np.asarray(times, dtype=np.float64),
        rps=np.asarray(rps, dtype=np.float64),
        n_mics=int(n_mics),
        n_harm=k_use,
        extra=dict(mp.spec.get("extra") or {}),
        **fields,
    )
    model = CombSpectrum(spec)
    with torch.no_grad():
        model.floor_mean_db.copy_(torch.as_tensor([float(params["floor_mean_db"])]))
        shape_db = np.asarray(params["floor_shape_db"], dtype=np.float64)
        ctrl = np.asarray(params.get("floor_ctrl_hz", model.ctrl_hz), dtype=np.float64)
        on_grid = np.interp(np.log2(model.ctrl_hz), np.log2(ctrl), shape_db)
        std = float(spec.floor_shape_std_db)
        z = torch.linalg.solve_triangular(
            model.shape_chol,
            torch.as_tensor(on_grid / std, dtype=model.shape_chol.dtype).reshape(-1, 1),
            upper=False,
        ).reshape(-1)
        model.floor_shape_z.copy_(z)
        model.floor_tilt_db_oct.copy_(
            torch.as_tensor([float(params.get("floor_tilt_db_oct", 0.0))])
        )
        prof = torch.zeros_like(model.profile_db)
        take_r = min(prof.shape[0], profile.shape[0])
        prof[:take_r, :k_use] = torch.as_tensor(profile[:take_r, :k_use], dtype=prof.dtype)
        model.profile_db.copy_(prof)
        g0 = np.atleast_1d(np.asarray(params["gamma0"], dtype=np.float64))
        sl = np.atleast_1d(np.asarray(params["gamma_slope"], dtype=np.float64))
        model.gamma0_raw.copy_(
            torch.as_tensor([_inv_softplus(v) for v in np.resize(g0, model.R)], dtype=prof.dtype)
        )
        model.slope_raw.copy_(
            torch.as_tensor([_inv_softplus(v) for v in np.resize(sl, model.R)], dtype=prof.dtype)
        )
        if spec.fit_width_power:
            ratio = float(params.get("width_power", spec.width_power)) / max(spec.width_power, 1e-9)
            model.log_width_power.copy_(torch.as_tensor([math.log(max(ratio, 1e-9))]))
        if spec.fit_coherence:
            k_half = float(params.get("coherence_k_half", spec.coherence_k_half))
            model.log_k_half.copy_(torch.as_tensor([math.log(max(k_half, 1e-3))]))
        if spec.fit_speed_law:
            model.amp_exp.copy_(
                torch.as_tensor([float(params.get("amp_exp", spec.amp_rps_exponent))])
            )
            model.floor_exp.copy_(
                torch.as_tensor([float(params.get("floor_exp", spec.amp_rps_exponent))])
            )
            model.floor_static_raw.copy_(
                torch.as_tensor([_inv_softplus(float(params.get("floor_static_rel", 0.0)))])
            )
        if spec.mic_floor and "mic_floor_db" in params:
            mf = np.asarray(params["mic_floor_db"], dtype=np.float64)
            model.mic_floor_db.copy_(
                torch.as_tensor(np.resize(mf, model.M), dtype=model.mic_floor_db.dtype)
            )
        if "mic_gain_db" in params:
            mg = np.atleast_2d(np.asarray(params["mic_gain_db"], dtype=np.float64))
            buf = np.zeros((model.M, model.R), dtype=np.float64)
            buf[: min(model.M, mg.shape[0]), : min(model.R, mg.shape[1])] = mg[: model.M, : model.R]
            model.mic_gain_db.copy_(torch.as_tensor(buf, dtype=model.mic_gain_db.dtype))
        m = model.forward().cpu().numpy().astype(np.float64)
    # ``gain_all_db`` is a rig-level composite, so CombSpectrum has no
    # parameter for it; it is a per-mic scalar and commutes with the window
    # kernel, so applying it here is exact.
    if params.get("gain_all_db") is not None:
        g = np.asarray(params["gain_all_db"], dtype=np.float64)
        g = np.resize(g, m.shape[0])
        m = m * (10.0 ** ((g - g.mean()) / 10.0))[:, None, None]
    return m * float(params.get("power_scale", 1.0))


@lru_cache(maxsize=8)
def _transfer_gain(n_bins: int, df: float, sample_rate_work: int) -> np.ndarray:
    """The render chain's power gain on an analysis grid, evaluated ONCE.

    Cached per (grid, work rate) so that the multiplication below happens at
    exactly one site with one buffer: double application is then structurally
    impossible rather than merely avoided (addendum 13 §67b).
    """
    from experiments.stochastic_fit.stage2 import render_transfer_power

    freqs = np.arange(int(n_bins), dtype=np.float64) * float(df)
    gain = np.asarray(
        render_transfer_power(freqs, sample_rate_work=int(sample_rate_work), sample_rate_out=SR),
        dtype=np.float64,
    )
    if gain.shape != freqs.shape:
        raise RuntimeError(f"render_transfer_power returned {gain.shape} for {freqs.shape} bins")
    gain.setflags(write=False)
    return gain


def predicted_m(
    mp: ModelParams,
    pg: Periodogram,
    *,
    n_mics: int,
    f_max: float | None = None,
    frame_chunk: int = 16,
    sample_rate_work: int = SAMPLE_RATE_WORK,
    apply_transfer: bool = True,
) -> np.ndarray:
    """``(M, N, F)`` LEGACY expected periodogram of ``mp`` on another window's grid.

    HISTORICAL-FORWARD BASELINE DIAGNOSTIC (addendum 4 §23, option 1 — the
    choice recorded in the proposals doc). This is the old descriptive model's
    OWN forward law: :class:`model.CombSpectrum` driven by the evaluation
    window's frame-mean carrier plus the known chirp-width covariate. It is
    **not** the exact moving-window kernel; the candidate's spectrum comes
    from ``revised_phase.predict_spectrum`` (mode ``"prior"``), which is, and
    the two are never described as the same observation law. The approximation
    is worst exactly where the contract says it is: a Michael ramp, where a
    frame-mean speed plus a widening term stands in for integrated moving
    atoms. Every record that carries a number from here also carries
    :data:`HISTORICAL_FORWARD_LABEL`.

    THE RENDER TRANSFER IS APPLIED EXACTLY ONCE, at the single site below, to
    the WHOLE spectrum — so the broadband floor and the comb lines carry the
    same ``render_transfer_power`` gain, which is the subtlest way the two
    families could otherwise disagree at the band edge (addendum 13 §67b): a
    floor missing the ~4.9 dB at 7900 Hz while the lines carry it would read as
    a baseline-vs-candidate difference rather than as a bug. The consequence to
    state wherever levels are reported: under this declared chain the export's
    ``profile_db`` is the SOURCE-side quantity of one observation law, never a
    calibrated absolute source spectrum. ``apply_transfer=False`` exists only
    so a test can measure the unfiltered reference.

    Clip-local latents are off — the drift GPs are pinned to zero standard
    deviation and the carrier correction to zero — which is what makes the
    prediction a function of the rig parameters only. No dense STFT covariance
    is built: the score is the marginal per-bin density on the fixed band.

    The frames are computed in blocks with one frame of context on each side,
    because the forward model keeps ``(rotors, orders, frames, bins)``
    intermediates: at the frozen 64 ms hop a whole 16 s window at once is
    about a gigabyte per intermediate. The context frame is what keeps the
    chirp-width term's central difference identical to the unchunked value at
    every kept frame.
    """
    band_top = float(f_max if f_max is not None else (mp.spec.get("f_max") or pg.freqs[-1]))
    freqs = np.asarray(pg.freqs, dtype=np.float64)
    times = np.asarray(pg.times, dtype=np.float64)
    rps = np.atleast_2d(np.asarray(pg.rps, dtype=np.float64))
    n = times.size
    step = max(int(frame_chunk), 1)
    out = np.empty((int(n_mics), n, freqs.size), dtype=np.float64)
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        c0, c1 = max(lo - 1, 0), min(hi + 1, n)
        block = _predict_block(
            mp,
            freqs,
            times[c0:c1],
            rps[:, c0:c1],
            n_mics=int(n_mics),
            band_top=band_top,
        )
        out[:, lo:hi] = block[:, lo - c0 : lo - c0 + (hi - lo)]
    if apply_transfer:
        # THE ONE SITE. Floor and lines are already summed here, so both carry
        # the gain exactly once, from one cached buffer.
        out *= _transfer_gain(freqs.size, float(pg.df), int(sample_rate_work))[None, None, :]
    return out


# ── the composite spectral risk ─────────────────────────────────────────────


@dataclass(frozen=True)
class FrameScore:
    """One window's per-frame marginal risk terms, with their absolute times.

    ``frame_times`` are the ANALYSIS-FRAME centres on the recording's own
    clock, which together with ``n_fft`` are the identity of the observation
    each term came from. ``window`` is provenance only — it no longer carries
    the exposure bookkeeping.
    """

    window: Window
    frame_times: np.ndarray  # (N,) s on the recording's own clock, frame centres
    frame_nll: np.ndarray  # (N,) sum over mics and in-band bins of I/M + log M
    n_cells_per_frame: int
    n_fft: int = OBS_N_FFT
    hop: int = OBS_HOP
    sr: int = SR


def marginal_frame_nll(
    power: np.ndarray, model: np.ndarray, band: np.ndarray
) -> tuple[np.ndarray, int]:
    """``(per-frame sum of I/M + log M, cells per frame)`` on the fixed band.

    The marginal per-bin exponential density, summed over microphones and
    in-band bins. Not a joint law over a redundant STFT — see the module
    docstring — and never divided by anything here.
    """
    m = np.maximum(np.asarray(model, dtype=np.float64), 1e-300)
    i = np.asarray(power, dtype=np.float64)
    cell = i / m + np.log(m)
    sel = np.asarray(band, dtype=bool)
    return cell[..., sel].sum(axis=(0, 2)), int(cell.shape[0] * int(sel.sum()))


def _frame_keys(item: FrameScore, *, quantum: float = 1e-4) -> list[tuple[str, int, int]]:
    """The identity of each analysis frame: ``(recording, centre, n_fft)``.

    The centre is quantized to 0.1 ms so that two copies of the same frame
    match exactly on an absolute epoch clock (DREGON's stamps sit near 1.5e9 s,
    where float64 resolves about 2e-7 s) while genuinely different frames — a
    64 ms hop apart — never collide.
    """
    rid = item.window.recording
    n_fft = int(item.n_fft)
    return [
        (rid, int(round(float(t) / quantum)), n_fft)
        for t in np.asarray(item.frame_times, dtype=np.float64)
    ]


def exposure_weights(items: Sequence[FrameScore]) -> list[np.ndarray]:
    """Positive per-ANALYSIS-FRAME exposure weights ``a_i``.

    Two factors, both explicit:

    * **the hop/window factor** ``hop / n_fft`` — a frame of ``n_fft`` samples
      taken every ``hop`` samples carries that fraction of its own window as
      new material. At the frozen convention it is ``1024 / 16384 = 1/16``,
      which is what makes the risk per unique second independent of the hop
      density instead of growing with the frame count. A fixed hop is a
      DECISION-RISK setting, not a statement of information content;
    * **duplicate splitting** — identical observations, keyed
      ``(recording, absolute frame centre, n_fft)``, share ONE unit of
      exposure between their copies. Duplicating a single frame entry, or a
      whole clip, therefore changes nothing.

    Exposure is never taken from clip spans: overlapping 8 s evaluation clips
    resolve into their analysis frames, and only frames that are the same
    observation are deduplicated.
    """
    counts: dict[tuple[str, int, int], int] = {}
    keyed = [_frame_keys(it) for it in items]
    for keys in keyed:
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
    out: list[np.ndarray] = []
    for it, keys in zip(items, keyed, strict=True):
        factor = float(it.hop) / float(it.n_fft)
        out.append(np.array([factor / counts[k] for k in keys], dtype=np.float64))
    return out


def unique_support_seconds(items: Sequence[FrameScore]) -> dict[str, float]:
    """Seconds of UNIQUE observed support per recording, and the total.

    One analysis frame owns the ``hop`` samples of new material its cadence
    advances by, so a set of unique frames covers ``n_unique * hop / sr``
    seconds. Duplicates add nothing, which is what keeps the normalization and
    the numerator consistent under duplication.
    """
    seen: set[tuple[str, int, int]] = set()
    per_rec: dict[str, float] = {}
    for it in items:
        step = float(it.hop) / float(it.sr)
        for k in _frame_keys(it):
            if k in seen:
                continue
            seen.add(k)
            per_rec[k[0]] = per_rec.get(k[0], 0.0) + step
    per_rec["__total__"] = float(sum(v for k, v in per_rec.items() if k != "__total__"))
    return per_rec


def composite_score(items: Sequence[FrameScore]) -> dict[str, Any]:
    """The frozen composite spectral RISK, total and per recording.

    ``sum_i a_i * (I_i / M_i + log M_i)`` with the positive exposure weights of
    :func:`exposure_weights`, normalized per unique observed support (nats per
    second of unique material). Lower is better. A COMPOSITE risk — proper for
    mean prediction — never an exact joint NLL, never evidence, never a
    percentage.
    """
    if not items:
        return dict(score=None, unique_seconds=0.0, per_recording={}, n_frames=0)
    weights = exposure_weights(items)
    unique = unique_support_seconds(items)
    per_rec_num: dict[str, float] = {}
    per_rec_frames: dict[str, int] = {}
    total = 0.0
    for it, w in zip(items, weights, strict=True):
        contribution = float(np.sum(np.asarray(it.frame_nll, dtype=np.float64) * w))
        rid = it.window.recording
        per_rec_num[rid] = per_rec_num.get(rid, 0.0) + contribution
        per_rec_frames[rid] = per_rec_frames.get(rid, 0) + int(np.size(it.frame_nll))
        total += contribution
    unique_total = float(unique["__total__"])
    per_rec = {
        rid: dict(
            score=per_rec_num[rid] / max(unique.get(rid, 0.0), 1e-12),
            weighted_nats=per_rec_num[rid],
            unique_seconds=unique.get(rid, 0.0),
            n_frames=per_rec_frames[rid],
        )
        for rid in sorted(per_rec_num)
    }
    cells = {int(it.n_cells_per_frame) for it in items}
    per_cell = (
        (total / max(unique_total, 1e-12)) / float(next(iter(cells))) if len(cells) == 1 else None
    )
    return dict(
        score=total / max(unique_total, 1e-12),
        weighted_nats=total,
        unique_seconds=unique_total,
        n_frames=int(sum(per_rec_frames.values())),
        n_items=len(items),
        cells_per_frame=sorted(cells),
        score_per_band_cell=per_cell,
        per_recording=per_rec,
        exposure=dict(
            granularity="analysis frame (recording, absolute centre, n_fft)",
            hop_window_factor=sorted({float(it.hop) / float(it.n_fft) for it in items}),
            n_fft=sorted({int(it.n_fft) for it in items}),
            hop=sorted({int(it.hop) for it in items}),
            note=(
                "a fixed hop is a decision-risk setting; the hop/window factor keeps the risk "
                "per unique second stable under hop density and must not be read as information "
                "content"
            ),
        ),
        form="positive exposure-weighted marginal I/M + logM, per unique observed second",
        units="nats/s of unique material (lower is better)",
        per_cell_note=(
            "score_per_band_cell is the same number divided by the constant cell count per "
            "frame, for readability only; the gate reads `score`"
        ),
    )


# ── the frozen scalar composite temperature (addendum 9 §45, §45a) ──────────

#: Stated in the record wherever ``T`` appears. It is exactly what the rule
#: estimates and nothing more.
TEMPERATURE_LABEL = (
    "scalar gain-direction calibration only: T matches the across-draw variance of the global "
    "log-power-gain score, and is NOT posterior calibration in every parameter direction"
)


def gain_frame_terms(
    power: np.ndarray, model: np.ndarray, band: np.ndarray
) -> tuple[np.ndarray, int]:
    """``(per-frame sum of (1 - I/M), cells per frame)`` on the fixed band.

    The gain-direction score of addendum 9 §45, summed over every in-band
    microphone and bin of a frame — the same cells, and later the same
    exposure weights, as the composite risk itself.
    """
    m = np.maximum(np.asarray(model, dtype=np.float64), 1e-300)
    i = np.asarray(power, dtype=np.float64)
    cell = 1.0 - i / m
    sel = np.asarray(band, dtype=bool)
    return cell[..., sel].sum(axis=(0, 2)), int(cell.shape[0] * int(sel.sum()))


def unnormalized_weighted_total(items: Sequence[FrameScore]) -> dict[str, float]:
    """``(sum_i a_i * term_i, H = sum_i a_i)`` on the UNNORMALIZED weights.

    The temperature rule must use the actual ``composite_risk`` weights
    ``a_i = hop / n_fft`` with the analysis-duplicate split — NOT the
    per-second-normalized reported score, which would make the priors wrong by
    duration (addendum 9 §45a). ``H`` counts every mic/bin/frame entry, so it
    is ``sum over frames of a_frame * cells_per_frame``.
    """
    weights = exposure_weights(items)
    total = 0.0
    h = 0.0
    for it, w in zip(items, weights, strict=True):
        total += float(np.sum(np.asarray(it.frame_nll, dtype=np.float64) * w))
        h += float(np.sum(w)) * float(it.n_cells_per_frame)
    return dict(total=total, exposure_h=h)


def unnormalized_gain_total(
    items: Sequence[FrameScore], draw_terms: Sequence[Sequence[np.ndarray]]
) -> dict[str, Any]:
    """``U_b = sum_i a_i * gain_term_i`` for each predictive draw, and ``H``.

    ``draw_terms[b]`` is a sequence of per-frame ``(1 - I_b/M)`` sums aligned
    with ``items``. The weights are the same hop/n_fft duplicate-split weights
    used by :func:`unnormalized_weighted_total`, so the temperature ``T = J/H``
    is consistent with the composite risk.
    """
    weights = exposure_weights(items)
    h = 0.0
    for it, w in zip(items, weights, strict=True):
        h += float(np.sum(w)) * float(it.n_cells_per_frame)
    u_by_draw: list[float] = []
    for terms_seq in draw_terms:
        total = 0.0
        for it, w, terms in zip(items, weights, terms_seq, strict=True):
            total += float(np.sum(np.asarray(terms, dtype=np.float64) * w))
        u_by_draw.append(total)
    return dict(u_by_draw=u_by_draw, exposure_h=h)


def composite_temperature(
    u_by_draw: Sequence[float],
    *,
    exposure_h: float,
    master_seed: int,
    clips: Sequence[str],
    rig: str,
    weighting: str = "unnormalized composite_risk weights a_i = hop/n_fft, duplicate-split",
) -> dict[str, Any]:
    """``T = J / H`` — the frozen scalar composite temperature.

    ``u_by_draw`` are the gain-direction scores ``U_b = sum_i a_i (1 - I_b/M_i)``
    of the legacy-baseline PREDICTIVE draws on the fixed calibration supports,
    with no per-draw gain normalization; ``exposure_h`` is ``H = sum_i a_i``
    over the same cells. ``J`` is the across-draw SAMPLE variance and
    ``T = J / H``; the fit then divides, i.e. uses ``L / T``.

    ONE convention globally: the unnormalized weights above. There is no
    second temperature, and nothing here is trainable or tunable after a
    candidate has been seen.

    ``J <= 0`` or a non-finite value FAILS the calibration — no clamp, no
    fallback: ``ok`` is False and ``temperature`` is ``None``, and the caller
    must refuse to freeze.
    """
    u = np.asarray(list(u_by_draw), dtype=np.float64)
    b = int(u.size)
    j = float(np.var(u, ddof=1)) if b >= 2 else float("nan")
    h = float(exposure_h)
    ok = bool(b >= 2 and np.isfinite(j) and j > 0.0 and np.isfinite(h) and h > 0.0)
    return dict(
        ok=ok,
        temperature=(float(j / h) if ok else None),
        J=j if np.isfinite(j) else None,
        H=h,
        B=b,
        master_seed=int(master_seed),
        rig=rig,
        calibration_clips=list(clips),
        u_by_draw=[float(v) for v in u],
        u_mean=(float(u.mean()) if b else None),
        weighting=weighting,
        application="the fit divides: L / T",
        label=TEMPERATURE_LABEL,
        failure=(
            ""
            if ok
            else "FAILED: J <= 0, non-finite, or fewer than two draws — no clamp and no fallback "
            "value is available, so the temperature must not be frozen"
        ),
        caveat=ADAPTIVE_SELECTION_CAVEAT,
    )


# ── the other two measured quantities ───────────────────────────────────────


def absolute_ltas_bands(x: np.ndarray, *, sr: int = SR, n: int = 8192) -> np.ndarray:
    """Band levels in dB with NO per-arm reference — the absolute LTAS.

    The same bands and the same Welch geometry as
    :func:`accept_stats.ltas_bands`, minus its self-reference to the clip's
    own 200-400 Hz mean. Self-referencing is per-arm normalization, which
    would let an arm launder a level error into an apparent shape match, so
    the absolute form is the primary quantity and the relative one is a
    secondary diagnostic (frozen addendum 3 section 17).
    """
    xx = np.asarray(x, dtype=np.float64)
    w = np.hanning(n + 1)[:n]
    starts = range(0, max(xx.size - n, 0), n // 2)
    if not len(starts):
        raise ValueError(f"{xx.size} samples is shorter than the {n}-point LTAS window")
    frames = np.stack([xx[s : s + n] * w for s in starts])
    power = (np.abs(np.fft.rfft(frames, axis=-1)) ** 2).mean(0)
    f = np.fft.rfftfreq(n, 1 / sr)
    db = 10.0 * np.log10(power + 1e-300)
    return np.array([db[(f >= lo) & (f < hi)].mean() for lo, hi in stats.BANDS])


def ltas_deviation_db(real: np.ndarray, arm: np.ndarray, *, mic: int = 0) -> dict[str, Any]:
    """Band-LTAS error of one arm against the real clip.

    PRIMARY: absolute-level error, no normalization of either arm. SECONDARY
    (diagnostic only): the shape error after each clip is referenced to its own
    200-400 Hz mean, i.e. ``accept_stats.ltas_bands``.
    """
    r = np.asarray(real, dtype=np.float64)[mic]
    a = np.asarray(arm, dtype=np.float64)[mic]
    abs_real, abs_arm = absolute_ltas_bands(r), absolute_ltas_bands(a)
    abs_dev = np.abs(abs_arm - abs_real)
    rel_dev = np.abs(stats.ltas_bands(a) - stats.ltas_bands(r))
    return dict(
        mean_abs_db=float(abs_dev.mean()),
        max_abs_db=float(abs_dev.max()),
        level_offset_db=float(np.mean(abs_arm - abs_real)),
        bands_real_db=[float(v) for v in abs_real],
        bands_arm_db=[float(v) for v in abs_arm],
        form="absolute-level LTAS error (no per-arm normalization)",
        shape_only_mean_abs_db=float(rel_dev.mean()),
        shape_only_max_abs_db=float(rel_dev.max()),
        shape_only_note="secondary diagnostic: each clip referenced to its own 200-400 Hz mean",
    )


# ── the pre-registered regime supports ──────────────────────────────────────


@dataclass(frozen=True)
class RegimeSupport:
    """The scored interval(s) of one regime inside one window.

    Read from the RAW TELEMETRY only, with the regime's frozen band: a sample
    is scored when every rotor is inside ``[min_rps, max_rps]`` there. The
    surrounding waveform is CONTEXT — the render, the STFT and the filter
    guard all need it — but it is never scored, and a mostly-cruise 8 s window
    is therefore never reported as a ramp MAE. Regime bands come from the
    manifest, never from a candidate and never from a refined track.
    """

    window: Window
    regime: str
    min_rps: float | None
    max_rps: float | None
    sample_mask: np.ndarray  # (T,) bool on the clip's own audio grid
    sr: int

    @property
    def n_scored(self) -> int:
        return int(self.sample_mask.sum())

    @property
    def scored_seconds(self) -> float:
        return float(self.n_scored / self.sr)

    def intervals(self) -> list[tuple[float, float]]:
        """Contiguous scored spans on the recording's own clock — provenance."""
        m = np.asarray(self.sample_mask, dtype=bool)
        if not m.any():
            return []
        padded = np.concatenate(([False], m, [False]))
        edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
        t0 = float(self.window.start_s)
        return [
            (t0 + float(a) / self.sr, t0 + float(b) / self.sr)
            for a, b in zip(edges[0::2], edges[1::2], strict=True)
        ]

    def as_dict(self) -> dict[str, Any]:
        return dict(
            window=self.window.key,
            regime=self.regime,
            min_rps=self.min_rps,
            max_rps=self.max_rps,
            n_samples_total=int(self.sample_mask.size),
            n_samples_scored=self.n_scored,
            scored_seconds=self.scored_seconds,
            scored_fraction=float(self.n_scored / max(self.sample_mask.size, 1)),
            intervals_on_recording_clock=[list(s) for s in self.intervals()],
            source="raw telemetry with the manifest's frozen regime band",
        )


def regime_support(
    window: Window,
    reference_rps: np.ndarray,
    *,
    regime: str,
    min_rps: float | None,
    max_rps: float | None,
    sr: int = SR,
) -> RegimeSupport:
    """The scored sample mask of ``regime`` inside ``window``, from raw telemetry."""
    ref = np.atleast_2d(np.asarray(reference_rps, dtype=np.float64))
    mask = np.ones(ref.shape[-1], dtype=bool)
    if min_rps is not None:
        mask &= np.all(ref >= float(min_rps), axis=0)
    if max_rps is not None:
        mask &= np.all(ref <= float(max_rps), axis=0)
    return RegimeSupport(window, regime, min_rps, max_rps, mask, int(sr))


def output_frame_centres(n_frames: int, n_samples: int, *, sr: int = SR) -> np.ndarray:
    """Clip-relative centres of a model's ``n_frames`` output frames.

    The tracker's own alignment, reused rather than re-invented:
    ``LayerPeakRPSMetric._unpack`` resamples the rotor label onto the model's
    frames with ``F.interpolate(..., align_corners=False)``, i.e. output frame
    ``j`` owns ``[j, j+1) * n_samples / n_frames``. Its centre is therefore
    ``(j + 0.5) * n_samples / n_frames``.
    """
    if n_frames <= 0:
        return np.zeros(0)
    return (np.arange(int(n_frames)) + 0.5) * (float(n_samples) / float(n_frames)) / float(sr)


def frames_in_support(
    n_frames: int, support: RegimeSupport | None, *, n_samples: int, sr: int = SR
) -> np.ndarray:
    """Boolean over output frames: does this frame's centre sit in the support?"""
    if support is None:
        return np.ones(int(n_frames), dtype=bool)
    centres = output_frame_centres(n_frames, n_samples, sr=sr)
    idx = np.clip((centres * sr).astype(int), 0, support.sample_mask.size - 1)
    return np.asarray(support.sample_mask, dtype=bool)[idx]


def pit_mae(
    score_fn: Callable[..., tuple[np.ndarray, np.ndarray, float]],
    fm: Any,
    metric: Any,
    audio: np.ndarray,
    rps_reference: np.ndarray,
    *,
    sr: int = SR,
    mics: Sequence[int],
    expected_samples: int,
    support: RegimeSupport | None = None,
    sample_tolerance: int = 0,
) -> dict[str, Any]:
    """PIT MAE of the frozen tracker, per microphone, on the SCORED support.

    ``score_fn`` is ``scripts/_synthetic_probe.py::score`` — injected rather
    than imported, so this module stays free of the scripts tree. The
    reference array is passed in by the caller and is the SAME raw telemetry
    for the real arm and for every synthetic arm.

    Two refusals, both of them gate integrity rather than convenience:

    * the arm must carry EXACTLY ``expected_samples`` samples (a
      ``sample_tolerance`` of 0 by default; any allowance is stated in the
      record) and exactly the frozen ``mics``. A short or narrow render is
      rejected — truncating it onto a shorter timeline, or averaging it over
      fewer microphones, is how a broken render scores well;
    * when a ``support`` is given, only output frames whose centres lie inside
      the pre-registered raw-telemetry interval are averaged. The mask, the
      scored fraction and the scored intervals travel in the record.
    """
    a = np.asarray(audio, dtype=np.float64)
    ref = np.asarray(rps_reference, dtype=np.float64)
    channels = [int(m) for m in mics]
    if a.ndim != 2:
        raise ValueError(f"arm audio must be (M, T), got shape {a.shape}")
    if abs(int(a.shape[-1]) - int(expected_samples)) > int(sample_tolerance):
        raise ValueError(
            f"arm carries {a.shape[-1]} samples, the frozen window is {expected_samples} "
            f"(tolerance {sample_tolerance}); refusing to score it on a different timeline"
        )
    if int(a.shape[0]) <= max(channels):
        raise ValueError(
            f"arm carries {a.shape[0]} microphones, the frozen set is {channels}; refusing to "
            "average over fewer microphones than the other arms"
        )
    if abs(int(ref.shape[-1]) - int(expected_samples)) > int(sample_tolerance):
        raise ValueError(
            f"reference carries {ref.shape[-1]} samples against the frozen {expected_samples}"
        )
    n = int(min(a.shape[-1], ref.shape[-1]))
    a, ref = a[:, :n], ref[:, :n]
    per_mic: list[float] = []
    pred0 = truth0 = None
    keep: np.ndarray | None = None
    for mic in channels:
        pred, truth, _mae = score_fn(fm, metric, a, ref, sr, mic)
        if keep is None:
            keep = frames_in_support(int(pred.shape[-1]), support, n_samples=n, sr=sr)
            if not bool(keep.any()):
                raise ValueError(
                    f"{(support.regime if support else 'window')} support selects no output "
                    f"frame of {n} samples: nothing to score, and a whole-window average is "
                    "not a substitute"
                )
        per_mic.append(float(np.abs(pred[:, keep] - truth[:, keep]).mean()))
        if pred0 is None:
            pred0, truth0 = pred, truth
    assert pred0 is not None and truth0 is not None and keep is not None
    return dict(
        mae=float(np.mean(per_mic)),
        per_mic=[float(v) for v in per_mic],
        mics=channels,
        spread=float(np.max(per_mic) - np.min(per_mic)),
        per_rotor=[float(v) for v in np.abs(pred0[:, keep] - truth0[:, keep]).mean(axis=1)],
        n_samples=int(n),
        n_output_frames=int(pred0.shape[-1]),
        n_scored_frames=int(keep.sum()),
        scored_fraction=float(keep.mean()),
        mean_reference_rps=float(np.mean(ref[:, support.sample_mask] if support else ref)),
        support=(support.as_dict() if support is not None else None),
        scored_frame_centres_s=[
            float(v)
            for v in (
                np.asarray(output_frame_centres(int(pred0.shape[-1]), n, sr=sr))[keep]
                + float(support.window.start_s if support else 0.0)
            )
        ],
        reference="raw telemetry (reference agreement, not proven ground truth)",
    )


# ── clustered statistics ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClusterInterval:
    """One-sided 95 % bounds on a mean over independent clusters.

    Both bounds are carried because the frozen gates need both directions: an
    improvement must have a lower bound above zero, and a paired composite
    delta must have an UPPER bound below zero.
    """

    n: int
    mean: float | None
    sem: float | None
    t_lower: float | None
    t_upper: float | None
    boot_lower: float | None
    boot_upper: float | None
    lower: float | None
    upper: float | None
    alpha: float
    note: str

    def as_dict(self) -> dict[str, Any]:
        return dict(
            n_clusters=self.n,
            mean=self.mean,
            sem=self.sem,
            t_lower=self.t_lower,
            t_upper=self.t_upper,
            bootstrap_lower=self.boot_lower,
            bootstrap_upper=self.boot_upper,
            lower=self.lower,
            upper=self.upper,
            alpha=self.alpha,
            rule=(
                "one-sided bounds at alpha, conservative: lower = min(t, cluster bootstrap), "
                "upper = max(t, cluster bootstrap)"
            ),
            caveat=ADAPTIVE_SELECTION_CAVEAT,
            note=self.note,
        )


def cluster_interval(
    values: Sequence[float], *, alpha: float = 0.05, n_boot: int = 20000, seed: int = 0
) -> ClusterInterval:
    """One-sided bounds on the mean, clustered at the unit supplied.

    A t bound (``df = n - 1``) and a cluster bootstrap percentile are both
    computed; the reported bounds are the CONSERVATIVE ones. Fewer than two
    clusters gives no bound at all — the caller must fail the gate, not assume
    one. Clusters are whatever the caller pairs on; for DREGON that is a whole
    recording (one independently recorded trajectory session), never a window,
    channel or render seed.
    """
    v = np.asarray(list(values), dtype=np.float64)
    if v.size == 0:
        return ClusterInterval(
            0, None, None, None, None, None, None, None, None, alpha, "no clusters"
        )
    mean = float(v.mean())
    if v.size < 2:
        return ClusterInterval(
            int(v.size),
            mean,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            alpha,
            "insufficient_clusters: a one-sided interval needs at least 2 independent clusters",
        )
    from scipy import stats as sps

    sem = float(v.std(ddof=1) / math.sqrt(v.size))
    half = float(sps.t.ppf(1.0 - alpha, v.size - 1) * sem)
    t_lower, t_upper = mean - half, mean + half
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, v.size, size=(int(n_boot), v.size))
    boot = v[draws].mean(axis=1)
    boot_lower = float(np.quantile(boot, alpha))
    boot_upper = float(np.quantile(boot, 1.0 - alpha))
    return ClusterInterval(
        int(v.size),
        mean,
        sem,
        t_lower,
        t_upper,
        boot_lower,
        boot_upper,
        float(min(t_lower, boot_lower)),
        float(max(t_upper, boot_upper)),
        alpha,
        "",
    )


# ── gates ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dict(name=self.name, passed=bool(self.passed), **self.detail)


def cohort_completeness_gate(
    *,
    required: Sequence[str],
    measured: Sequence[str],
    support_reports: Sequence[Mapping[str, Any]],
    name: str,
) -> Gate:
    """The frozen cohort must be COMPLETE before anything is bootstrapped.

    Missing coverage is a stop condition, not a smaller cohort: the gate fails
    when a frozen recording produced no paired measurement, or when any
    support report came back ``sufficient=False``. "Report it, never pad or
    repeat" means the run fails, not that the cohort shrinks to whatever
    happened to work.
    """
    need = [str(r) for r in required]
    have = sorted({str(r) for r in measured})
    missing = [r for r in need if r not in have]
    extra = [r for r in have if r not in need]
    insufficient = [
        dict(recording=r.get("recording"), regime=r.get("regime"), note=r.get("note"))
        for r in support_reports
        if not bool(r.get("sufficient", True))
    ]
    checks = dict(
        cohort_complete=not missing,
        no_unexpected_recordings=not extra,
        every_support_sufficient=not insufficient,
    )
    return Gate(
        name=name,
        passed=all(checks.values()),
        detail=dict(
            checks=checks,
            required_recordings=need,
            measured_recordings=have,
            missing_recordings=missing,
            unexpected_recordings=extra,
            insufficient_supports=insufficient,
            rule="the frozen cohort is required in full; no intersection shrink, no padding",
        ),
    )


def dregon_pit_gate(
    per_recording: Mapping[str, Mapping[str, float]],
    *,
    gap_fraction: float,
    alpha: float,
    required_recordings: Sequence[str],
    seed: int = 0,
) -> Gate:
    """The DREGON cruise PIT-MAE gate, clustered at whole-recording level.

    ``per_recording[rec]`` carries ``real``, ``baseline`` and ``candidate``
    MAEs of the frozen tracker against the SAME raw telemetry array, each read
    on that regime's pre-registered support. Four conditions, all required:

    * every recording of the FROZEN cohort is present — a partial cohort fails
      here, before any bootstrap, instead of being accepted as "at least two
      clusters";
    * the real-vs-synthetic gap is positive with a one-sided 95 % bound
      (the calibration this gate is defined against);
    * the point target ``E_new <= E_real + f * (E_best_synth - E_real)``, on
      the cluster means;
    * the paired improvement ``E_best_synth - E_new`` excludes "no
      improvement" at a one-sided 95 % bound clustered by recording.

    Clusters are independently recorded trajectory sessions. Nothing here is
    computed from rounded or median numbers: every value is a paired
    whole-recording measurement at full precision.
    """
    need = [str(r) for r in required_recordings]
    recs = sorted(per_recording)
    missing = [r for r in need if r not in recs]
    rows = []
    for r in recs:
        d = per_recording[r]
        e_real, e_base, e_new = float(d["real"]), float(d["baseline"]), float(d["candidate"])
        target = e_real + float(gap_fraction) * (e_base - e_real)
        rows.append(
            dict(
                recording=r,
                real=e_real,
                baseline=e_base,
                candidate=e_new,
                target=target,
                gap=e_base - e_real,
                improvement=e_base - e_new,
                margin=target - e_new,
            )
        )
    gap = cluster_interval([r["gap"] for r in rows], alpha=alpha, seed=seed)
    imp = cluster_interval([r["improvement"] for r in rows], alpha=alpha, seed=seed)
    mean_new = float(np.mean([r["candidate"] for r in rows])) if rows else float("nan")
    mean_target = float(np.mean([r["target"] for r in rows])) if rows else float("nan")
    checks = dict(
        cohort_complete=bool(need and not missing),
        gap_positive=bool(gap.lower is not None and gap.lower > 0.0),
        point_target=bool(rows and mean_new <= mean_target),
        improvement_excludes_zero=bool(imp.lower is not None and imp.lower > 0.0),
        enough_clusters=bool(len(rows) >= 2),
    )
    return Gate(
        name="dregon_cruise_pit_mae",
        passed=all(checks.values()),
        detail=dict(
            checks=checks,
            gap_fraction=float(gap_fraction),
            required_recordings=need,
            missing_recordings=missing,
            mean_candidate=mean_new,
            mean_target=mean_target,
            clusters=rows,
            cluster_unit="independently recorded trajectory session (whole recording)",
            gap_interval=gap.as_dict(),
            improvement_interval=imp.as_dict(),
            regime="cruise only — no standby/ramp acceptance for DREGON",
        ),
    )


def composite_delta_gate(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    alpha: float,
    tolerance: float | None = 0.0,
    seed: int = 0,
    name: str = "held_out_composite_delta",
) -> Gate:
    """Held-out composite spectral risk: the paired delta must be negative.

    Frozen form (addendum 3 section 19): the paired mean
    ``delta = candidate - baseline`` must be ``< tolerance`` AND its one-sided
    95 % UPPER bound must be ``< tolerance``, clustered at whole-recording
    level. Absolute nats per unique second — never a percentage. ``tolerance``
    comes from the frozen manifest/calibration; ``0.0`` is plain
    non-regression and a ``None`` tolerance fails the gate.
    """
    if tolerance is None:
        return missing_tolerance_gate(
            name,
            what="composite-risk non-regression tolerance (nats/s)",
            reason="the frozen record carries no composite tolerance for this cohort/regime",
        )
    recs = sorted(set(baseline) & set(candidate))
    rows = [
        dict(
            recording=r,
            baseline=float(baseline[r]),
            candidate=float(candidate[r]),
            delta=float(candidate[r]) - float(baseline[r]),
        )
        for r in recs
    ]
    interval = cluster_interval([r["delta"] for r in rows], alpha=alpha, seed=seed)
    mean_delta = float(np.mean([r["delta"] for r in rows])) if rows else float("nan")
    checks = dict(
        mean_delta_negative=bool(rows and mean_delta < float(tolerance)),
        upper_bound_negative=bool(interval.upper is not None and interval.upper < float(tolerance)),
        enough_clusters=bool(len(rows) >= 2),
        fully_paired=bool(len(recs) == len(baseline) == len(candidate)),
    )
    return Gate(
        name=name,
        passed=all(checks.values()),
        detail=dict(
            checks=checks,
            tolerance=float(tolerance),
            mean_delta=mean_delta,
            clusters=rows,
            interval=interval.as_dict(),
            score="composite (overlap-exposure weighted marginal I/M + logM per unique second)",
            direction="lower is better; delta = candidate - baseline",
            unpaired_recordings=dict(
                baseline_only=sorted(set(baseline) - set(candidate)),
                candidate_only=sorted(set(candidate) - set(baseline)),
            ),
        ),
    )


def missing_tolerance_gate(name: str, *, what: str, reason: str) -> Gate:
    """A gate that FAILS because its frozen tolerance is absent.

    A ``None`` tolerance never waves a gate through: the check is recorded as
    failed with the reason, the run exits nonzero, and no permissive default
    is invented (frozen addendum 3 §20, addendum 6 §32).
    """
    return Gate(
        name=name,
        passed=False,
        detail=dict(
            checks=dict(tolerance_available=False),
            what=what,
            reason=reason,
            rule="a missing null-variation tolerance fails loudly; it is never defaulted",
        ),
    )


def ltas_gate(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    tolerance_db: float | None,
    name: str = "baseline_calibrated_ltas",
    unit: str = "recording",
) -> Gate:
    """Baseline-calibrated LTAS gate on the ABSOLUTE-level band error.

    Point non-regression against the baseline's own error, plus a tolerance
    that must come from the frozen null-variation calibration (the baseline's
    own variability — for Michael's, BLOCK variation conditional on FLY124),
    never from an absolute preregistered dB number — those are known to fail
    real-against-real on this rig — and never tuned after a candidate has been
    seen. A ``None`` tolerance fails the gate.
    """
    if tolerance_db is None:
        return missing_tolerance_gate(
            name,
            what="absolute-level LTAS non-regression tolerance (dB)",
            reason=(
                "the calibration record carries no null-variation LTAS tolerance for this "
                "cohort/regime — re-run --prepare with at least two frozen render seeds"
            ),
        )
    keys = sorted(set(baseline) & set(candidate))
    rows = [
        dict(
            unit=k,
            baseline_mean_abs_db=float(baseline[k]),
            candidate_mean_abs_db=float(candidate[k]),
            delta_db=float(candidate[k]) - float(baseline[k]),
        )
        for k in keys
    ]
    mean_base = float(np.mean([r["baseline_mean_abs_db"] for r in rows])) if rows else float("nan")
    mean_cand = float(np.mean([r["candidate_mean_abs_db"] for r in rows])) if rows else float("nan")
    checks = dict(
        not_worse=bool(rows and mean_cand <= mean_base + float(tolerance_db)),
        fully_paired=bool(keys and len(keys) == len(baseline) == len(candidate)),
    )
    return Gate(
        name=name,
        passed=all(checks.values()),
        detail=dict(
            checks=checks,
            tolerance_db=float(tolerance_db),
            tolerance_source="frozen null-variation calibration",
            pairing_unit=unit,
            mean_baseline_db=mean_base,
            mean_candidate_db=mean_cand,
            metric="absolute-level LTAS band error vs the real clip (mean |dB|)",
            units=rows,
        ),
    )


def conditional_non_regression_gate(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    tolerance: float | None,
    alpha: float,
    seed: int = 0,
    name: str,
    quantity: str,
    recording: str,
) -> Gate:
    """Non-regression of a paired quantity over BLOCKS of one recording.

    Michael's addendum-3 §19 form: the gate is conditional on FLY124, paired
    per held-out block, with the tolerance taken from that recording's own
    block variation. The delta's one-sided upper bound is reported and
    required, and every statement is explicitly conditional — never an
    across-recording population claim. A ``None`` tolerance fails the gate.
    """
    if tolerance is None:
        return missing_tolerance_gate(
            name,
            what=f"{quantity} non-regression tolerance, conditional on {recording}",
            reason=(
                f"the calibration record carries no block-variation tolerance for {quantity} "
                f"on {recording} — re-run --prepare with at least two frozen render seeds and "
                "at least two held-out blocks"
            ),
        )
    keys = sorted(set(baseline) & set(candidate))
    rows = [
        dict(
            block=k,
            baseline=float(baseline[k]),
            candidate=float(candidate[k]),
            delta=float(candidate[k]) - float(baseline[k]),
        )
        for k in keys
    ]
    interval = cluster_interval([r["delta"] for r in rows], alpha=alpha, seed=seed)
    mean_delta = float(np.mean([r["delta"] for r in rows])) if rows else float("nan")
    checks = dict(
        mean_within_tolerance=bool(rows and mean_delta <= float(tolerance)),
        upper_bound_within_tolerance=bool(
            interval.upper is not None and interval.upper <= float(tolerance)
        ),
        fully_paired=bool(keys and len(keys) == len(baseline) == len(candidate)),
    )
    return Gate(
        name=name,
        passed=all(checks.values()),
        detail=dict(
            checks=checks,
            quantity=quantity,
            tolerance=float(tolerance),
            tolerance_source=f"block variation within {recording}",
            mean_delta=mean_delta,
            blocks=rows,
            interval=interval.as_dict(),
            conditioning=(
                f"conditional on {recording} blocks only — the across-recording population "
                "criterion was waived and no population claim is made"
            ),
        ),
    )


def michaels_ratio_gate(
    blocks: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    ratio_max: float,
    alpha: float,
    seed: int = 0,
    recording: str = "FLY124",
) -> Gate:
    """Michael's fixed-recording gate on FLY124: ratio of equally weighted MAEs.

    The frozen arithmetic (addendum 3 section 22) is a RATIO OF MEANS, not a
    mean of ratios::

        E_rig_arm = (E_standby_arm + E_ramp_arm + E_cruise_arm) / 3
        gate: E_rig_candidate / E_rig_baseline <= ratio_max

    A mean of per-regime ratios would give a tiny standby MAE disproportionate
    leverage and invites a near-zero denominator. Every regime and its own
    ratio are still reported separately. Uncertainty is block variation
    CONDITIONAL on this one recording; no across-recording population claim is
    made or implied.
    """
    per_regime: dict[str, Any] = {}
    for regime in sorted(blocks):
        rows = [
            dict(
                block=str(b["key"]),
                baseline=float(b["baseline"]),
                candidate=float(b["candidate"]),
                ratio=float(b["candidate"]) / float(b["baseline"]),
            )
            for b in blocks[regime]
        ]
        base = float(np.mean([r["baseline"] for r in rows])) if rows else float("nan")
        cand = float(np.mean([r["candidate"] for r in rows])) if rows else float("nan")
        # upper bound on the block-level delta, conditional on this recording
        delta = cluster_interval(
            [r["candidate"] - r["baseline"] for r in rows], alpha=alpha, seed=seed
        )
        per_regime[regime] = dict(
            n_blocks=len(rows),
            baseline_mae=base,
            candidate_mae=cand,
            ratio=(cand / base if base else float("nan")),
            blocks=rows,
            conditional_block_delta_interval=delta.as_dict(),
            conditioning=f"conditional on {recording} blocks — not a population claim",
        )
    regimes = sorted(per_regime)
    base_maes = [per_regime[r]["baseline_mae"] for r in regimes]
    cand_maes = [per_regime[r]["candidate_mae"] for r in regimes]
    finite = bool(regimes) and all(np.isfinite(base_maes)) and all(np.isfinite(cand_maes))
    e_base = float(np.mean(base_maes)) if finite else float("nan")
    e_cand = float(np.mean(cand_maes)) if finite else float("nan")
    aggregate = e_cand / e_base if finite and e_base else float("nan")
    checks = dict(
        all_regimes_present=bool(finite and len(regimes) == len(blocks)),
        aggregate_within=bool(np.isfinite(aggregate) and aggregate <= float(ratio_max)),
    )
    return Gate(
        name="michaels_fly124_mae_ratio",
        passed=all(checks.values()),
        detail=dict(
            checks=checks,
            recording=recording,
            ratio_max=float(ratio_max),
            regimes=regimes,
            rig_baseline_mae=e_base,
            rig_candidate_mae=e_cand,
            aggregate_ratio=aggregate,
            arithmetic="ratio of equally weighted per-regime MAEs (NOT a mean of ratios)",
            per_regime=per_regime,
        ),
    )


# ── loading real windows ────────────────────────────────────────────────────


def assert_raw_reference(rps_key: str) -> str:
    """Refuse a refined/posterior track as the primary scoring reference."""
    key = str(rps_key)
    if key == REFINED_RPS_KEY or key == "auto":
        raise ValueError(
            f"scoring reference {key!r} is not raw telemetry: the primary reference must be the "
            f"same raw telemetry array for the real and synthetic arms (one of {RAW_RPS_KEYS}). "
            "A refined label is fit provenance, not a scoring reference."
        )
    if key not in RAW_RPS_KEYS:
        raise ValueError(f"unknown scoring reference {key!r}; expected one of {RAW_RPS_KEYS}")
    return key


def load_window(
    window: Window,
    *,
    dataset: str,
    version: str | None,
    channels: str | tuple[int, ...] | None,
    rps_key: str,
    target_sr: int = SR,
) -> Clip:
    """One real window at ``target_sr``, cut on the recording's own clock.

    The rotor track attached is ``rps_key``: for the scoring arm that is RAW
    telemetry, which is the reference BOTH arms are measured against and the
    trajectory the synthetic arms are driven by.
    """
    clip = C.load_clip(
        dataset,
        window.recording,
        float(window.start_s),
        float(window.duration_s),
        version=version,
        channels=channels,
        rps_key=rps_key,
        clip_id=f"{window.clip_id}_{rps_key}",
    )
    return C.decimate(clip, target_sr)


def window_periodogram(clip: Clip, *, n_fft: int = OBS_N_FFT, hop: int = OBS_HOP) -> Periodogram:
    """The FROZEN observation: periodic Hann, NFFT 16384, 64 ms hop.

    The window and the hop are the shared kernel's own constants
    (:mod:`phase_kernel`), and ``data.periodogram`` builds the identical
    periodic Hann. At this hop the frames overlap 16-fold — about 110 frames
    in an 8 s clip — which is why the score is a composite RISK over marginal
    terms rather than a joint likelihood, and why the exposure bookkeeping is
    carried by the analysis frames themselves (:func:`exposure_weights`),
    never by the clip spans.
    """
    return periodogram(clip, n_fft=n_fft, hop=hop)


def observation_band(
    freqs: np.ndarray, *, f_min: float = OBS_F_MIN, f_max: float = OBS_F_MAX
) -> np.ndarray:
    """The fixed 30-7900 Hz band mask, identical for every arm.

    Fixed by the protocol, never narrowed to hide a filter's passband loss.
    """
    f = np.asarray(freqs, dtype=np.float64)
    return (f >= float(f_min)) & (f <= float(f_max))


def frame_times_on_clock(window: Window, pg: Periodogram) -> np.ndarray:
    """Frame centres on the RECORDING's clock, whatever the clock's origin."""
    return float(window.start_s) + np.asarray(pg.times, dtype=np.float64)


def adapter_agreement(
    pred: np.ndarray, observed_mean: np.ndarray, band: np.ndarray
) -> dict[str, Any]:
    """Does a predicted-``M`` adapter match its own renderer's mean statistics?

    ``observed_mean`` is the mean periodogram over independent render draws of
    the SAME parameters. The level offset and the SHAPE residual after removing
    it are reported separately, because they fail for different reasons: the
    offset is a units/level disagreement and is only meaningful when the render
    ran on the physical path (``normalize_rms=None`` plus
    :func:`to_renderer_units`); under the renderer's default RMS normalization
    it is arbitrary. This is a verification of the adapter, not a gate.
    """
    p = np.maximum(np.asarray(pred, dtype=np.float64), 1e-300)
    o = np.maximum(np.asarray(observed_mean, dtype=np.float64), 1e-300)
    sel = np.asarray(band, dtype=bool)
    pd, od = 10.0 * np.log10(p[..., sel]), 10.0 * np.log10(o[..., sel])
    offset = float(np.mean(od - pd))
    resid = od - pd - offset
    return dict(
        level_offset_db=offset,
        shape_mean_abs_db=float(np.mean(np.abs(resid))),
        shape_median_abs_db=float(np.median(np.abs(resid))),
        shape_p95_abs_db=float(np.quantile(np.abs(resid), 0.95)),
        band_cells=int(pd.size),
        note=(
            "level_offset_db is a real disagreement only if the render used normalize_rms=None "
            "with the declared conversion; with the renderer's default RMS normalization the "
            "offset is arbitrary and only the shape residual is comparable"
        ),
    )


def json_ready(obj: Any) -> Any:
    """Full-precision JSON: arrays to lists, numpy scalars to python, no rounding."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(json_ready(payload), indent=1))
    return p


__all__ = [
    "ADAPTIVE_SELECTION_CAVEAT",
    "ANALYSIS_GRID_HZ",
    "CLIP_LOCAL_LATENTS",
    "HISTORICAL_FORWARD_LABEL",
    "HISTORICAL_RATE_FACTOR_DB",
    "OBS_F_MAX",
    "OBS_F_MIN",
    "OBS_HOP",
    "OBS_N_FFT",
    "RAW_RPS_KEYS",
    "REFINED_RPS_KEY",
    "RENDER_RATE_FACTOR_DB",
    "SAMPLE_RATE_WORK",
    "SCHEMA",
    "TEMPERATURE_LABEL",
    "CandidateExport",
    "ClusterInterval",
    "ExportBundle",
    "FamilyCoverage",
    "FrameScore",
    "Gate",
    "ModelParams",
    "RegimeSupport",
    "SupportReport",
    "Window",
    "absolute_ltas_bands",
    "adapter_agreement",
    "aggregate_nuisance",
    "artifact_digest",
    "assert_raw_reference",
    "check_explicit_windows",
    "cluster_interval",
    "cohort_completeness_gate",
    "composite_delta_gate",
    "composite_score",
    "composite_temperature",
    "conditional_non_regression_gate",
    "dregon_pit_gate",
    "export_family",
    "exposure_weights",
    "family_coverage",
    "frame_times_on_clock",
    "frames_in_support",
    "gain_frame_terms",
    "import_provenance",
    "legacy_profile_db_to_candidate",
    "json_ready",
    "load_window",
    "ltas_deviation_db",
    "ltas_gate",
    "marginal_frame_nll",
    "michaels_ratio_gate",
    "missing_tolerance_gate",
    "observation_band",
    "output_frame_centres",
    "pit_mae",
    "predicted_m",
    "rate_factor_db",
    "read_candidate_export",
    "read_export",
    "regime_intervals",
    "regime_support",
    "resolve_evaluation_windows",
    "resolve_support_windows",
    "select_family",
    "to_renderer_units",
    "training_leakage",
    "union_seconds",
    "unique_support_seconds",
    "unnormalized_gain_total",
    "unnormalized_weighted_total",
    "window_periodogram",
    "write_json",
]
