"""The three FROZEN noise-model-v2 round gates, as pure functions.

Everything here is arithmetic on measurements handed in by
``scripts/noise_v2_round_score.py``: no loading, no rendering, no scoring. The
protocol these gates re-enact is the previous campaign's, read out of the code
that produced the recorded numbers rather than restated from memory. What
follows is that protocol, with the exact sources.

The probe protocol, reproduced from the previous campaign
========================================================

**Runner.** ``scripts/stochastic_fit_revised_eval.py --prepare`` (baseline
calibration) then ``--check`` (candidate), driven by a frozen manifest;
``docs/revised-phase-c4-eval-manifest-v2.json`` is the last one. R1 re-enacts
the ``--check`` measurement path for a v2 candidate arm, against the frozen
BASELINE numbers recorded in
``docs/experiments/revised-phase-campaign.qmd:331-358`` (the calibration file
itself, ``results/revised_phase/baseline_v2/calibration.json``, is under the
gitignored ``results/`` tree and is not in the repository).

**Scorer.** ``Tracker`` (``scripts/stochastic_fit_revised_eval.py:827-859``)
resolves experiment ``hppnet_l2_r2_s0``, ckpt ``best``, records its SHA-256
(``6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1``), loads
it with ``zoo.load``, pairs it with ``metrics.salience_layers.LayerPeakRPSMetric``
and injects ``scripts/_synthetic_probe.py::score`` into
``revised_eval.pit_mae``. ``score`` (``scripts/_synthetic_probe.py:95-127``)
runs ONE microphone through the frame model, reads peaks off
``logsigmoid(layers)`` with ``peak_readout``, interpolates the rotor label onto
the model's own output frames, and resolves rotor identity by the PIT
permutation that minimises the mean absolute error.

**PIT MAE.** ``revised_eval.pit_mae`` (``:2044-2133``) refuses an arm that does
not carry exactly the frozen sample count and microphone set, averages the
per-microphone MAEs (``mics = range(n_mics)``, i.e. ALL EIGHT microphones of
both rigs in the frozen runs), and averages only the output frames whose
centres fall inside the pre-registered raw-telemetry regime support
(``regime_support``, ``:2000-2016``; ``frames_in_support``, ``:2033-2041``).
The rotor reference is the RAW telemetry of the recording for the real arm and
for every synthetic arm (``motors_command`` on DREGON, ``rps`` on Michael's).

**Supports.** DREGON: the five room-2 no-source cruise windows, 4 s each,
regime band ``rps >= 65``, explicitly listed in the frozen manifest and
mirrored in ``scripts/noise_v2_likelihood_window.py:105-111``. Michael's:
FLY124 ``@8+8`` and ``@16+8`` standby (20-45 rev/s), ``@27.68+8`` ramp
(45-65 rev/s, ~0.99 s of scored support inside an 8 s context window) and
``@40+8`` and ``@56+8`` cruise (>= 65 rev/s)
(``noise_v2_likelihood_window.py:115-121``, resolved by the frozen evaluator
from the auto/support window rules of the manifest cohort).

**Render seeds and aggregation.** ``--check`` renders every arm at the frozen
``null_variation.seeds = [2001, 2002, 2003, 2004]``
(``docs/revised-phase-c4-eval-manifest-v2.json:31-44``;
``measure_arm_set``/``run_check``). ``per_recording`` averages over that
recording's windows AND render seeds (``:1119-1124``); ``per_window`` averages
over render seeds only (``:1127-1132``). DREGON clusters at whole recording
(one 4 s window each, so cluster == window); Michael's blocks at window.

**DREGON cruise gate.** ``revised_eval.dregon_pit_gate`` (``:2295-2370``) with
``gates.dregon_gap_fraction = 0.7``: per recording ``target = E_real + 0.7 *
(E_baseline - E_real)``, and the recording-level one-sided 95 % interval comes
from ``cluster_interval`` (``:2181-2235``): a Student-t bound with ``df = n-1``
AND a 20000-draw cluster bootstrap at ``seed = gates.bootstrap_seed = 0``,
reported CONSERVATIVELY (``lower = min``, ``upper = max``). Frozen v2 baseline
numbers: real ``1.2187080817581784``, baseline synthetic
``2.1877858830655468``, hence target ``1.897062542673336``.

**Michael's gate.** ``revised_eval.michaels_ratio_gate`` (``:2642-2720``): a
RATIO OF EQUALLY WEIGHTED per-regime MAEs, never a mean of per-regime ratios —
``E_rig = (E_standby + E_ramp + E_cruise) / 3`` with each regime's MAE the mean
over its blocks, and the gate is ``E_rig_candidate / E_rig_baseline <= 1.05``.
Frozen v2 baseline per regime: standby ``0.3171026880710335``, ramp
``8.007098121439082``, cruise ``0.7557833849952398``; equal-regime mean
``3.026661398168452``; deterministic 1.05x bound ``3.177994468076875``.

**How the un-fitted Michael's regimes are supplied.** The old campaign had no
fitted standby/ramp model for FLY124: the frozen ``baseline_map``
(``docs/revised-phase-c4-eval-manifest-v2.json:260-279``) applies the FLY125
cruise export's aggregate to the FLY124 RAMP as a declared cross-recording
extrapolation, and the FLY125 standby export's aggregate to FLY124 standby, and
in both cases the render is driven by the REAL FLY124 raw ``rps`` carrier of
that support through the renderer's own trajectory sampler (``ArmModel.render``,
``scripts/stochastic_fit_revised_eval.py:485-531``; the sampler's realised
shaft speed is reported as a DIAGNOSTIC only, ``:1035-1047``). v2 keeps the
same route with one simplification that follows from having no standby fit at
all: the FLY125 cruise fit drives all three Michael's regimes, rendered on each
support's real carrier (approved decision 3,
``docs/explainers/noise-model-v2-plan.qmd:1544-1545``).

**Proxy.** ``ltas_abs_db`` is ``revised_eval.ltas_deviation_db(real, arm,
mic=0)["mean_abs_db"]`` (``:1914-1939``): 8192-point Welch band levels in dB
with NO per-arm normalisation, absolute-level error, mic 0, taken on the
support clip as a whole (``scripts/noise_v2_spectrogram_proxy.py:331-370``).
The approved R1 threshold is closure 0.70 of the C3-to-oracle gap measured in
``results/noise_v2/criteria/proxy.json``: DREGON cruise
``2.4307927323931553 - 0.70 * (2.4307927323931553 - 1.784816176956543) =
1.9786091435875266`` dB over the five supports, Michael's cruise
``1.5333406982093933 - 0.70 * (1.5333406982093933 - 1.0852372760667532) =
1.2196683027095454`` dB over the two cruise supports. ``mr_ltas``
(``noise_v2_spectrogram_proxy.py:198-238``) is reported, never decisive.

**Likelihood.** Composite spectral risk at NFFT 2048 / hop 512:
``revised_eval.marginal_frame_nll`` + ``FrameScore`` + ``composite_score``
(``:1606-1753``) — positive exposure-weighted ``sum a_i (I_i/M_i + log M_i)``
per unique observed second, ``a_i = hop/n_fft`` with duplicate frames splitting
one unit of exposure. The R1 comparison is against the speed-matched stationary
oracle of ``results/noise_v2/short_whittle/short_whittle.json`` (arm
``oracle_np``): the Welch MEAN periodogram of a legal disjoint real segment of
the SAME recording, start chosen on a 0.25 s grid to minimise the per-rotor
mean-speed mismatch, broadcast over the scored support's frames
(``scripts/noise_v2_short_whittle.py:311-332``;
``noise_v2_likelihood_window.oracle_mean_periodogram``). Approved decision 6
pools Michael's FLY124 CRUISE supports and reports per band with the comb band
(>= 300 Hz) decisive and the floor band (< 300 Hz) reported; R1 freezes the
margin and passes when the model is strictly below the oracle on the comb band.

Two protocol details had to be RE-DERIVED rather than reproduced, and both are
recorded in every gate payload:

* the DREGON gate's paired ``E_baseline - E_candidate`` improvement interval
  needs the frozen PER-RECORDING baseline MAEs, which live only in the
  gitignored ``calibration.json``. R1 therefore gates on the one-sided 95 %
  UPPER bound of the candidate's own recording-level mean against the frozen
  scalar target — the same one-sided 95 % recording-level instrument, applied
  to the candidate mean instead of to the paired difference. ``hppnet_gate``
  computes the paired interval as well whenever per-recording baselines are
  supplied;
* the oracle of the likelihood gate is recorded in ``short_whittle.json`` over
  the FULL 30-7900 Hz band only, so the per-band (300 Hz split) oracle risk is
  recomputed here from the same oracle periodogram. The full-band recomputation
  is cross-checked against the recorded pooled number and the delta is
  reported.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.stochastic_fit import revised_eval as RE

__all__ = [
    "DREGON_CRUISE_SUPPORTS",
    "MICHAELS_SUPPORTS",
    "ScoredSupport",
    "SpectralCell",
    "hppnet_gate",
    "likelihood_gate",
    "proxy_gate",
]


# ── the frozen supports ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScoredSupport:
    """One frozen scored support: the identity every gate pairs on."""

    rig: str
    recording: str
    start_s: float
    duration_s: float
    regime: str

    @property
    def window(self) -> RE.Window:
        return RE.Window(self.recording, float(self.start_s), float(self.duration_s), self.regime)

    @property
    def key(self) -> str:
        return self.window.key

    @property
    def min_rps(self) -> float | None:
        return REGIME_BANDS[self.regime][0]

    @property
    def max_rps(self) -> float | None:
        return REGIME_BANDS[self.regime][1]

    def as_dict(self) -> dict[str, Any]:
        return dict(
            rig=self.rig,
            recording=self.recording,
            start_s=float(self.start_s),
            duration_s=float(self.duration_s),
            regime=self.regime,
            key=self.key,
        )


#: The frozen regime bands, read from the manifest cohorts (and mirrored in
#: ``scripts/noise_v2_likelihood_window.py:150-154``).
REGIME_BANDS: dict[str, tuple[float, float | None]] = {
    "standby": (20.0, 45.0),
    "ramp": (45.0, 65.0),
    "cruise": (65.0, None),
}

#: The five DREGON room-2 cruise scoring windows of the frozen v2 manifest.
DREGON_CRUISE_SUPPORTS: tuple[ScoredSupport, ...] = (
    ScoredSupport("dregon", "free-flight_nosource_room2", 1512727397.2050455, 4.0, "cruise"),
    ScoredSupport("dregon", "hovering_nosource_room2", 1511903905.3944898, 4.0, "cruise"),
    ScoredSupport("dregon", "updown_nosource_room2", 1511903578.348311, 4.0, "cruise"),
    ScoredSupport("dregon", "rectangle_nosource_room2", 1511905725.952559, 4.0, "cruise"),
    ScoredSupport("dregon", "spinning_nosource_room2", 1511905200.978012, 4.0, "cruise"),
)

#: Michael's five FLY124 supports the frozen evaluator resolved.
MICHAELS_SUPPORTS: tuple[ScoredSupport, ...] = (
    ScoredSupport("michaels", "FLY124", 8.0, 8.0, "standby"),
    ScoredSupport("michaels", "FLY124", 16.0, 8.0, "standby"),
    ScoredSupport("michaels", "FLY124", 27.68, 8.0, "ramp"),
    ScoredSupport("michaels", "FLY124", 40.0, 8.0, "cruise"),
    ScoredSupport("michaels", "FLY124", 56.0, 8.0, "cruise"),
)

MICHAELS_CRUISE_SUPPORTS: tuple[ScoredSupport, ...] = tuple(
    s for s in MICHAELS_SUPPORTS if s.regime == "cruise"
)

#: The rotor track every arm is SCORED against, per rig (never a refined label).
RAW_RPS_KEY: dict[str, str] = {"dregon": "motors_command", "michaels": "rps"}
DATASET: dict[str, str] = {"dregon": "DREGON-frames", "michaels": "michaels-frames"}

#: The frozen render seeds of ``null_variation.seeds``.
RENDER_SEEDS: tuple[int, ...] = (2001, 2002, 2003, 2004)

#: The frozen scorer.
SCORER_EXPERIMENT = "hppnet_l2_r2_s0"
SCORER_CKPT = "best"
SCORER_SHA256 = "6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1"


# ── the frozen numbers the gates are defined against ────────────────────────

#: Frozen v2 DREGON cruise baseline calibration
#: (``docs/experiments/revised-phase-campaign.qmd:343-348``).
DREGON_REAL_PIT_MAE = 1.2187080817581784
DREGON_BASELINE_PIT_MAE = 2.1877858830655468
#: ``gates.dregon_gap_fraction`` of the frozen manifest: the candidate must
#: close 30 % of the real-to-synthetic gap.
DREGON_GAP_FRACTION = 0.7
DREGON_PIT_TARGET = 1.897062542673336

#: Frozen v2 Michael's baseline per-regime MAEs and their equal-regime mean
#: (``docs/experiments/revised-phase-campaign.qmd:350-354``).
MICHAELS_BASELINE_PIT_MAE: dict[str, float] = {
    "standby": 0.3171026880710335,
    "ramp": 8.007098121439082,
    "cruise": 0.7557833849952398,
}
MICHAELS_BASELINE_REGIME_MEAN = 3.026661398168452
MICHAELS_RATIO_MAX = 1.05
MICHAELS_PIT_BOUND = 3.177994468076875

#: Approved proxy gate: closure 0.70 of the C3-to-oracle ``ltas_abs_db`` gap
#: measured in ``results/noise_v2/criteria/proxy.json``.
PROXY_CLOSURE = 0.70
PROXY_REFERENCE: dict[str, dict[str, float]] = {
    "dregon_cruise": dict(
        oracle_db=1.784816176956543,
        c3_db=2.4307927323931553,
        current_best_db=2.126453600529959,
        oracle_spread_db=1.141525996979129,
        n_supports=5.0,
    ),
    "michaels_cruise": dict(
        oracle_db=1.0852372760667532,
        c3_db=1.5333406982093933,
        current_best_db=1.3320148816302133,
        oracle_spread_db=0.05340504126022878,
        n_supports=2.0,
    ),
}
PROXY_LTAS_GATE_DB: dict[str, float] = {
    group: ref["c3_db"] - PROXY_CLOSURE * (ref["c3_db"] - ref["oracle_db"])
    for group, ref in PROXY_REFERENCE.items()
}

#: Likelihood front end and band split (approved decisions 6 and 7).
LIKELIHOOD_N_FFT = 2048
LIKELIHOOD_HOP = 512
BAND_EDGE_HZ = 300.0

ALPHA = 0.05
BOOTSTRAP_SEED = 0


# ── the HPPNet gate ─────────────────────────────────────────────────────────


def _present(supports: Sequence[ScoredSupport], values: Mapping[str, float]) -> list[str]:
    return [s.key for s in supports if s.key not in values]


def hppnet_gate(
    pred_by_support: Mapping[str, float],
    real_by_support: Mapping[str, float],
    *,
    baseline_by_support: Mapping[str, float] | None = None,
    alpha: float = ALPHA,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """The frozen HPPNet probe gate: DREGON cruise MAE and Michael's ratio.

    ``pred_by_support`` and ``real_by_support`` map a frozen support key
    (``Window.key``) to the frozen tracker's PIT MAE in rev/s, already averaged
    over all eight microphones and over the frozen render seeds, and read on
    that support's pre-registered raw-telemetry regime support — i.e. exactly
    the ``per_window`` quantity of ``scripts/stochastic_fit_revised_eval.py``.
    The real arm is measured on the same supports and is a PROTOCOL CHECK, not
    a gate input: the gate's real/baseline numbers are the frozen scalars of
    the v2 calibration.

    DREGON: the clusters are the five room-2 recordings (one 4 s window each),
    so the recording-level cluster mean and its one-sided 95 % upper bound come
    straight from ``revised_eval.cluster_interval``. The gate requires the
    complete cohort, the point mean at or below ``DREGON_PIT_TARGET``, and the
    one-sided upper bound at or below the same target.

    Michael's: per-regime MAE is the mean over that regime's blocks, the rig
    MAE is the equal-weight mean of the three regimes, and the gate is the
    ratio of means against ``MICHAELS_RATIO_MAX`` times the frozen baseline
    equal-regime mean.
    """
    dregon = _dregon_pit(pred_by_support, real_by_support, baseline_by_support, alpha, seed)
    michaels = _michaels_ratio(pred_by_support, real_by_support, alpha, seed)
    return dict(
        name="hppnet_pit_mae",
        scorer=dict(experiment=SCORER_EXPERIMENT, ckpt=SCORER_CKPT, sha256=SCORER_SHA256),
        dregon_cruise=dregon,
        michaels_fly124=michaels,
        **{"pass": bool(dregon["pass"] and michaels["pass"])},
    )


def _dregon_pit(
    pred: Mapping[str, float],
    real: Mapping[str, float],
    baseline: Mapping[str, float] | None,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    missing = _present(DREGON_CRUISE_SUPPORTS, pred)
    clusters: list[dict[str, Any]] = [
        dict(
            recording=s.recording,
            support=s.key,
            candidate=float(pred[s.key]),
            real=(float(real[s.key]) if s.key in real else None),
            baseline=(
                float(baseline[s.key]) if baseline is not None and s.key in baseline else None
            ),
        )
        for s in DREGON_CRUISE_SUPPORTS
        if s.key in pred
    ]
    values: list[float] = [float(c["candidate"]) for c in clusters]
    interval = RE.cluster_interval(values, alpha=alpha, seed=seed)
    mean = float(np.mean(values)) if values else float("nan")
    upper = interval.upper
    reals: list[float] = [float(c["real"]) for c in clusters if c["real"] is not None]
    mean_real = float(np.mean(reals)) if reals else None
    paired = None
    if baseline is not None:
        deltas: list[float] = [
            float(c["baseline"]) - float(c["candidate"])
            for c in clusters
            if c["baseline"] is not None
        ]
        if deltas:
            paired = RE.cluster_interval(deltas, alpha=alpha, seed=seed).as_dict()
    checks = dict(
        cohort_complete=not missing,
        enough_clusters=len(clusters) >= 2,
        point_target=bool(values and mean <= DREGON_PIT_TARGET),
        interval_upper_within_target=bool(upper is not None and upper <= DREGON_PIT_TARGET),
    )
    return dict(
        checks=checks,
        target_rev_s=DREGON_PIT_TARGET,
        gap_fraction=DREGON_GAP_FRACTION,
        frozen_real_rev_s=DREGON_REAL_PIT_MAE,
        frozen_baseline_rev_s=DREGON_BASELINE_PIT_MAE,
        mean_candidate_rev_s=mean,
        candidate_interval=interval.as_dict(),
        margin_rev_s=(DREGON_PIT_TARGET - mean if values else None),
        clusters=clusters,
        cluster_unit="independently recorded trajectory session (one 4 s cruise window each)",
        missing_supports=missing,
        real_arm=dict(
            mean_rev_s=mean_real,
            frozen_rev_s=DREGON_REAL_PIT_MAE,
            delta=(None if mean_real is None else mean_real - DREGON_REAL_PIT_MAE),
            relative=(
                None
                if mean_real is None
                else abs(mean_real - DREGON_REAL_PIT_MAE) / DREGON_REAL_PIT_MAE
            ),
            role="protocol reproduction check on the real arm, never a gate input",
        ),
        paired_improvement_interval=paired,
        paired_improvement_note=(
            "the frozen per-recording baseline MAEs live only in the gitignored "
            "results/revised_phase/baseline_v2/calibration.json; without them the paired "
            "E_baseline - E_candidate interval of revised_eval.dregon_pit_gate cannot be "
            "reproduced, so the decisive one-sided 95 % instrument is applied to the "
            "candidate's own recording-level mean against the frozen scalar target"
        ),
        **{"pass": bool(all(checks.values()))},
    )


def _michaels_ratio(
    pred: Mapping[str, float],
    real: Mapping[str, float],
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    missing = _present(MICHAELS_SUPPORTS, pred)
    per_regime: dict[str, Any] = {}
    for regime in sorted({s.regime for s in MICHAELS_SUPPORTS}):
        blocks: list[dict[str, Any]] = [
            dict(
                support=s.key,
                candidate=float(pred[s.key]),
                real=(float(real[s.key]) if s.key in real else None),
            )
            for s in MICHAELS_SUPPORTS
            if s.regime == regime and s.key in pred
        ]
        vals: list[float] = [float(b["candidate"]) for b in blocks]
        cand = float(np.mean(vals)) if vals else float("nan")
        base = MICHAELS_BASELINE_PIT_MAE[regime]
        delta = (
            RE.cluster_interval([v - base for v in vals], alpha=alpha, seed=seed).as_dict()
            if len(vals) >= 2
            else None
        )
        per_regime[regime] = dict(
            n_blocks=len(blocks),
            candidate_mae=cand,
            baseline_mae=base,
            ratio=(cand / base if base else float("nan")),
            blocks=blocks,
            conditional_block_delta_interval=delta,
            conditioning="conditional on FLY124 blocks — not a population claim",
        )
    regimes = sorted(per_regime)
    cand_maes = [per_regime[r]["candidate_mae"] for r in regimes]
    finite = bool(regimes) and all(np.isfinite(cand_maes))
    rig_cand = float(np.mean(cand_maes)) if finite else float("nan")
    ratio = rig_cand / MICHAELS_BASELINE_REGIME_MEAN if finite else float("nan")
    checks = dict(
        cohort_complete=not missing,
        all_regimes_present=bool(finite and set(regimes) == set(MICHAELS_BASELINE_PIT_MAE)),
        aggregate_within=bool(np.isfinite(ratio) and ratio <= MICHAELS_RATIO_MAX),
    )
    return dict(
        checks=checks,
        recording="FLY124",
        arithmetic="ratio of equally weighted per-regime MAEs (NOT a mean of ratios)",
        ratio_max=MICHAELS_RATIO_MAX,
        rig_candidate_mae=rig_cand,
        rig_baseline_mae=MICHAELS_BASELINE_REGIME_MEAN,
        aggregate_ratio=ratio,
        point_bound_rev_s=MICHAELS_PIT_BOUND,
        margin_rev_s=(MICHAELS_PIT_BOUND - rig_cand if finite else None),
        per_regime=per_regime,
        missing_supports=missing,
        unfitted_regimes=dict(
            regimes=["standby", "ramp"],
            route=(
                "the FLY125 cruise fit rendered on each support's real FLY124 raw rps carrier "
                "through the renderer's trajectory sampler (approved decision 3); the old "
                "campaign's equivalent was the manifest baseline_map cross-recording "
                "extrapolation of the FLY125 exports"
            ),
        ),
        **{"pass": bool(all(checks.values()))},
    )


# ── the spectrogram proxy gate ──────────────────────────────────────────────


def proxy_gate(
    real_audio: Mapping[str, np.ndarray],
    synth_audio: Mapping[str, np.ndarray],
    *,
    mic: int = 0,
    mr_ltas_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> dict[str, Any]:
    """The approved ``ltas_abs_db`` proxy gate, with ``mr_ltas`` reported.

    ``real_audio`` and ``synth_audio`` map a frozen support key to ``(M, T)``
    audio in the evaluator's absolute units; the statistic reads ``mic`` (0)
    only, over the whole support clip, exactly as the criteria study did.
    ``mr_ltas_fn`` is injected — the multi-resolution variant lives in
    ``scripts/noise_v2_spectrogram_proxy.py`` and this module never imports the
    scripts tree — and is reported, never decisive.

    Grouping follows the approved decision: DREGON cruise over the five room-2
    supports (stated as a mean with its spread, because the oracle spread there
    is larger than the whole gap it gates) and Michael's cruise over the two
    FLY124 cruise supports.
    """
    groups = dict(
        dregon_cruise=DREGON_CRUISE_SUPPORTS,
        michaels_cruise=MICHAELS_CRUISE_SUPPORTS,
    )
    out: dict[str, Any] = dict(
        name="proxy_ltas_abs_db",
        primary="ltas_abs_db (revised_eval.ltas_deviation_db mean_abs_db, mic 0, absolute level)",
        secondary="mr_ltas (reported only)",
        closure=PROXY_CLOSURE,
        mic=int(mic),
        groups={},
    )
    passed = True
    for group, supports in groups.items():
        rows: list[dict[str, Any]] = []
        for s in supports:
            if s.key not in real_audio or s.key not in synth_audio:
                continue
            real = np.atleast_2d(np.asarray(real_audio[s.key], dtype=np.float64))
            arm = np.atleast_2d(np.asarray(synth_audio[s.key], dtype=np.float64))
            n = min(int(real.shape[-1]), int(arm.shape[-1]))
            dev = RE.ltas_deviation_db(real[:, :n], arm[:, :n], mic=int(mic))
            rows.append(
                dict(
                    support=s.key,
                    regime=s.regime,
                    ltas_abs_db=float(dev["mean_abs_db"]),
                    ltas_max_abs_db=float(dev["max_abs_db"]),
                    level_offset_db=float(dev["level_offset_db"]),
                    shape_only_mean_abs_db=float(dev["shape_only_mean_abs_db"]),
                    mr_ltas=(
                        None
                        if mr_ltas_fn is None
                        else float(mr_ltas_fn(real[int(mic), :n], arm[int(mic), :n]))
                    ),
                )
            )
        missing = [s.key for s in supports if s.key not in {r["support"] for r in rows}]
        vals = [r["ltas_abs_db"] for r in rows]
        mean = float(np.mean(vals)) if vals else float("nan")
        spread = float(max(vals) - min(vals)) if len(vals) >= 2 else 0.0 if vals else float("nan")
        mr_vals = [r["mr_ltas"] for r in rows if r["mr_ltas"] is not None]
        gate_db = PROXY_LTAS_GATE_DB[group]
        checks = dict(
            cohort_complete=not missing,
            within_gate=bool(vals and mean <= gate_db),
        )
        passed = passed and all(checks.values())
        out["groups"][group] = dict(
            checks=checks,
            gate_db=gate_db,
            reference=PROXY_REFERENCE[group],
            mean_ltas_abs_db=mean,
            spread_ltas_abs_db=spread,
            margin_db=(gate_db - mean if vals else None),
            mean_mr_ltas=(float(np.mean(mr_vals)) if mr_vals else None),
            n_supports=len(rows),
            missing_supports=missing,
            supports=rows,
            **{"pass": bool(all(checks.values()))},
        )
    out["pass"] = bool(passed)
    return out


# ── the likelihood gate ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class SpectralCell:
    """One support's observed periodogram with the model and oracle means.

    ``power``, ``model`` and ``oracle`` are ``(M, N, F)`` on the SAME frozen
    front end (``LIKELIHOOD_N_FFT`` / ``LIKELIHOOD_HOP``); ``frame_times_s``
    are clip-relative analysis-frame centres, which the gate puts on the
    recording's own clock for the exposure bookkeeping.
    """

    support: ScoredSupport
    freqs_hz: np.ndarray
    frame_times_s: np.ndarray
    power: np.ndarray
    model: np.ndarray
    oracle: np.ndarray


def likelihood_gate(
    cells: Sequence[SpectralCell],
    *,
    oracle_json: Mapping[str, Any] | None = None,
    band_edge_hz: float = BAND_EDGE_HZ,
    f_min: float = RE.OBS_F_MIN,
    f_max: float = RE.OBS_F_MAX,
    n_fft: int = LIKELIHOOD_N_FFT,
    hop: int = LIKELIHOOD_HOP,
) -> dict[str, Any]:
    """Composite risk against the speed-matched stationary oracle, per band.

    Pooled over the frozen Michael's FLY124 CRUISE supports at NFFT 2048 /
    hop 512, split at ``band_edge_hz`` into the decisive comb band
    (``>= 300 Hz``) and the reported floor band (``< 300 Hz``). PASS in R1 is
    the model strictly below the oracle on the comb band; the margin
    ``model - oracle`` is what R1 freezes for later rounds.

    ``oracle_json`` is the parsed ``short_whittle.json``: it carries the
    recorded FULL-band pooled ``oracle_np`` risk, which is used as a
    reproduction cross-check of the recomputed oracle (the recorded file has no
    per-band split, so the per-band oracle is recomputed here from the same
    oracle periodogram).
    """
    wanted = {s.key for s in MICHAELS_CRUISE_SUPPORTS}
    seen = {c.support.key for c in cells}
    missing = sorted(wanted - seen)
    unexpected = sorted(seen - wanted)
    bands = _band_masks(cells, f_min=f_min, f_max=f_max, band_edge_hz=band_edge_hz)
    per_band: dict[str, Any] = {}
    for band_name, mask_of in bands.items():
        arms: dict[str, Any] = {}
        for arm in ("model", "oracle"):
            items: list[RE.FrameScore] = []
            for cell in cells:
                mask = mask_of(cell)
                nll, n_cells = RE.marginal_frame_nll(cell.power, getattr(cell, arm), mask)
                items.append(
                    RE.FrameScore(
                        cell.support.window,
                        float(cell.support.start_s) + np.asarray(cell.frame_times_s, float),
                        nll,
                        n_cells,
                        n_fft=int(n_fft),
                        hop=int(hop),
                    )
                )
            score = RE.composite_score(items)
            arms[arm] = dict(
                score=score["score"],
                score_per_band_cell=score["score_per_band_cell"],
                weighted_nats=score["weighted_nats"],
                unique_seconds=score["unique_seconds"],
                n_frames=score["n_frames"],
                cells_per_frame=score["cells_per_frame"],
            )
        margin = (
            None
            if arms["model"]["score"] is None or arms["oracle"]["score"] is None
            else float(arms["model"]["score"] - arms["oracle"]["score"])
        )
        per_band[band_name] = dict(
            band_hz=_band_edges(band_name, f_min=f_min, f_max=f_max, band_edge_hz=band_edge_hz),
            model_nats_per_s=arms["model"]["score"],
            oracle_nats_per_s=arms["oracle"]["score"],
            margin_nats_per_s=margin,
            below_oracle=bool(margin is not None and margin < 0.0),
            arms=arms,
        )
    comb = per_band["comb"]
    checks = dict(
        cohort_complete=not missing and not unexpected,
        comb_below_oracle=bool(comb["below_oracle"]),
    )
    out: dict[str, Any] = dict(
        name="likelihood_composite_risk",
        front_end=dict(n_fft=int(n_fft), hop=int(hop), sr=RE.SR, window="periodic Hann"),
        band_hz=[float(f_min), float(f_max)],
        band_edge_hz=float(band_edge_hz),
        decisive_band="comb",
        pooled_over=[c.support.key for c in cells],
        missing_supports=missing,
        unexpected_supports=unexpected,
        checks=checks,
        per_band=per_band,
        frozen_margin_nats_per_s=comb["margin_nats_per_s"],
        frozen_margin_note=(
            "R1 freezes the comb-band model-minus-oracle margin; later rounds are held to it"
        ),
        oracle=dict(
            arm="oracle_np",
            definition=(
                "Welch mean periodogram of a speed-matched legal disjoint real segment of the "
                "same recording, broadcast over the scored support's frames"
            ),
            source="results/noise_v2/short_whittle/short_whittle.json",
        ),
        units="nats/s of unique observed material (lower is better)",
        **{"pass": bool(all(checks.values()))},
    )
    out["oracle"]["reproduction"] = _oracle_reproduction(cells, oracle_json, n_fft=n_fft, hop=hop)
    return out


def _band_masks(
    cells: Sequence[SpectralCell], *, f_min: float, f_max: float, band_edge_hz: float
) -> dict[str, Callable[[SpectralCell], np.ndarray]]:
    def full(cell: SpectralCell) -> np.ndarray:
        return RE.observation_band(np.asarray(cell.freqs_hz, float), f_min=f_min, f_max=f_max)

    def comb(cell: SpectralCell) -> np.ndarray:
        return full(cell) & (np.asarray(cell.freqs_hz, float) >= float(band_edge_hz))

    def floor(cell: SpectralCell) -> np.ndarray:
        return full(cell) & (np.asarray(cell.freqs_hz, float) < float(band_edge_hz))

    return dict(comb=comb, floor=floor, full=full)


def _band_edges(band: str, *, f_min: float, f_max: float, band_edge_hz: float) -> list[float]:
    if band == "comb":
        return [float(band_edge_hz), float(f_max)]
    if band == "floor":
        return [float(f_min), float(band_edge_hz)]
    return [float(f_min), float(f_max)]


def _oracle_reproduction(
    cells: Sequence[SpectralCell],
    oracle_json: Mapping[str, Any] | None,
    *,
    n_fft: int,
    hop: int,
) -> dict[str, Any]:
    """Cross-check the recomputed FULL-band oracle against the recorded pooled one.

    ``short_whittle.json`` records the oracle per support (``rows``) as a risk
    in nats/s plus that support's ``weighted_nats`` and ``unique_seconds``. The
    frozen supports are pairwise disjoint, so no analysis frame is duplicated
    between them and the pooled risk over any subset is exactly
    ``sum(weighted_nats) / sum(unique_seconds)`` — which is how the recorded
    reference for this subset is derived.
    """
    if oracle_json is None:
        return dict(available=False, reason="no short_whittle.json supplied")
    keys = {c.support.key for c in cells}
    rows = [
        r
        for r in (oracle_json.get("rows") or [])
        if str(r.get("arm")) == "oracle_np"
        and int(r.get("n_fft", -1)) == int(n_fft)
        and int(r.get("hop", -1)) == int(hop)
        and str(r.get("support")) in keys
        and bool(r.get("available"))
    ]
    if len(rows) != len(keys):
        return dict(
            available=False,
            reason=(
                f"short_whittle.json carries {len(rows)} oracle rows at {n_fft}/{hop} for the "
                f"{len(keys)} pooled supports"
            ),
            supports=sorted(keys),
        )
    nats = float(sum(float(r["weighted_nats"]) for r in rows))
    seconds = float(sum(float(r["unique_seconds"]) for r in rows))
    return dict(
        available=True,
        recorded_pooled_nats_per_s=nats / seconds if seconds else None,
        recorded_unique_seconds=seconds,
        n_rows=len(rows),
        schema=oracle_json.get("schema"),
        git_head=(oracle_json.get("provenance") or {}).get("git_head"),
        band_hz=oracle_json.get("band_hz"),
        note=(
            "recorded numbers are FULL-band only; compare against per_band.full.oracle_nats_per_s"
        ),
    )


def relative_delta(a: float, b: float) -> float:
    """``|a - b| / |b|`` — the reproduction tolerance the smoke test reports."""
    return math.inf if b == 0.0 else abs(float(a) - float(b)) / abs(float(b))
