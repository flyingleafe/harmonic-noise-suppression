"""The two control streams the fitting campaign is measured against.

Both are built from the accepted fitted policy so that everything OUTSIDE the
noise family — sample rate, clip length, speech corpus, SNR prior, the
augmentation block, the silence arm, the trajectory generator, the reuse
factors — is byte-identical to the run under test. A transfer difference can
then only come from the family, which is the only reason a control is worth the
GPU hours.

C1, the SIMPLE control (``ctrl_static.yaml``)
    Static combs. Amplitudes are frozen for the clip, the shaft does not
    wander, no line diffuses, the floor does not breathe. What is NOT
    simplified is the timbre: real combs differ rotor to rotor, put their
    energy in the blade-passing orders, and fall off with order, so the family
    samples roll-off, blade count, blade-pass emphasis, a boost on the
    blade-passing fundamental alone, per-order irregularity and a rotor-to-
    rotor spread — widely, and around both fitted rigs rather than only near
    them. Two extra arms render the fitted DREGON and Michael's profiles
    themselves with the dynamics switched off, so the family provably CONTAINS
    the measured timbres and the question is only whether the dynamics matter.

    This control decides whether the whole stochastic apparatus is load
    bearing. If a static comb transfers, the apparatus is not needed and that
    is the best possible outcome; the campaign's own claim is that it will not,
    because a real high-order comb is broadened and amplitude modulated, and a
    model trained on razor-thin static lines learns a cue that real audio does
    not carry.

C2, the DIVERSE-PARAMETER test (``ctrl_diverse.yaml``)
    Not a control — a hypothesis test. The fitted presets sample a tight
    neighbourhood of two rigs; this samples the same MODEL over a wide
    parameter space, keeping only those quantities the measurements actually
    pin: the line-width regime, the shaft-versus-per-harmonic split of the
    phase noise, the amplitude process, the speed laws, and the label error.
    Timbre is sampled wide. The hypothesis is that a model forced to
    disentangle a broad distribution learns the disentangling operation rather
    than one rig's fingerprint, and therefore transfers if the family is a
    faithful description of real rotor noise.

    The failure mode this must avoid is the original stochastic-comb stream,
    which was diverse in the wrong way: it sampled regions no real rig occupies
    (lines so wide the comb merged, floors that buried every order, amplitudes
    that came and went within a clip) and was simply a harder, different task.
    Every range below is therefore anchored to a measurement, and two guards
    keep a draw learnable at all — the per-rotor floor coverage guard, and a
    width prior whose upper end is the widest the flight fits ever asked for
    rather than the widest the renderer can do.

WHAT IS DELIBERATELY THE SAME IN BOTH. The static per-rotor shaft-minus-label
offset stays on. It is a property of the DATA, not of the noise model: the
telemetry is not the shaft, measured at a robust 0.41 rev/s on Michael's crops
and 1.38 on DREGON's. Rendering the comb exactly on the label would hand C1 a
precision no real recording carries, and the control would then win or lose for
the wrong reason.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

#: The timbre family, shared by both controls so that C1 and C2 differ ONLY in
#: their dynamics. Every entry is wider than either rig's fitted value and
#: brackets both.
#:
#: ``rolloff_p``            fitted 1.6-2.1 (DREGON), flat (Michael's); real
#:                          single-rotor combs measure 0.6-1.6 on the bench
#: ``blade_counts``         two- and three-blade rotors. The blade-passing
#:                          orders are the MULTIPLES of the blade count, so on
#:                          Michael's two-blade rotors they are the even orders
#: ``blade_emphasis_db``    fitted 0.5-3.0 (DREGON), 13 (Michael's)
#: ``bpf_boost_db``         the blade-passing FUNDAMENTAL alone; Michael's rig
#:                          fit puts order 2 some 16-19 dB over its neighbours
#: ``rotor_similarity``     1.0 is four rotors of one timbre (DREGON: the
#:                          per-rotor medians span 0.3 dB), and 0.3 is four
#:                          nearly independent ones (Michael's span 18.9 dB)
#: ``harm_jitter_db``       per-order irregularity, fitted 4.5-7.0
PROFILE_FAMILY: dict[str, Any] = {
    "rolloff_p": [0.3, 2.2],
    "harm_jitter_db": [1.5, 8.0],
    "blade_counts": [2, 3],
    "blade_emphasis_db": [2.0, 16.0],
    "bpf_boost_db": [0.0, 14.0],
    "rotor_similarity": [0.3, 1.0],
    "rotor_delta_std_db": [0.0, 4.0],
    # The floor's SHAPE is drawn, not preset: a control that inherited one
    # rig's measured floor curve would be testing that curve, not the family.
    "floor_shape_std_db": [2.0, 9.0],
    "floor_shape_oct": [0.7, 3.0],
    "floor_tilt_db_oct": [-9.0, -1.0],
    # How far the floor sits under the typical line peak. Both rigs measure
    # -22 to -2 dB, which is why so many high orders wash out.
    "floor_rel_db": [-22.0, -2.0],
    # A stopped rotor is not digital silence: the chain's own floor measures
    # 0.175 of a cruise clip on the frozen split, 0.370 on a ramp.
    "floor_static_rel": [0.05, 0.40],
    # THE LEARNABILITY GUARD. The pooled fraction is satisfiable with one rotor
    # buried under the other three, whose speed is then a label with no
    # evidence in the clip. Every rotor must keep a fifth of its in-band lines.
    "min_lines_above_floor": 0.30,
    "min_lines_above_floor_per_rotor": 0.20,
    # Per-microphone structure as a POPULATION, never a fitted vector pinned to
    # microphone indices — index carries no measured physics, and the fitted
    # rigs' index-locked vectors are what made channel 0 of DREGON score twice
    # the error of the full array.
    "mic_gain_all_db": [0.0, 6.0],
    "mic_floor_std_db": [0.0, 3.0],
}

#: The label error, in both controls. A property of the telemetry, not of the
#: noise family: robust per-(clip, rotor) scale 0.41 rev/s on Michael's
#: training crops, 1.38 on DREGON's.
LABEL_ERROR: dict[str, Any] = {"shaft_offset_rps": [0.4, 1.4]}

#: C1: every stochastic element off. A line is then a pure tone and the clip's
#: spectrum is one fixed picture.
#:
#: ``gamma0_hz`` is NOT the rendered width in ``line_mode: fm`` — the width
#: there comes from the shaft jitter and the phase diffusion, both zero here.
#: It survives because the floor calibration compares the floor against line
#: PEAKS, and a zero-width line has an infinite one; 1 Hz is the analysis
#: window's own half width, so the calibration reads a window-limited tone.
STATIC_OFF: dict[str, Any] = {
    "gamma0_hz": [1.0, 1.0],
    "gamma_slope_hz": [0.0, 0.0],
    "shaft_jitter_rps": [0.0, 0.0],
    "shaft_jitter_log_std": [0.0, 0.0],
    "phase_diffusion_hz_per_order": [0.0, 0.0],
    "harm_gp_std_db": [0.0, 0.0],
    "harm_coherence": [0.0, 0.0],
    "floor_gp_std_db": [0.0, 0.0],
    "floor_tilt_gp_std": [0.0, 0.0],
    "umod_std_db": [0.0, 0.0],
}

#: C2: the dynamics, sampled wide but anchored.
#:
#: ``shaft_jitter_rps``  the coherent, width-proportional-to-k term. The bench
#:                       recordings put the rig's own shaft at 0.019 rev/s
#:                       (width law 0.023 Hz/order, and single lines on
#:                       Motor1_70 cross -3 dB at 0.14 Hz at k=4 and 1.87 at
#:                       k=32); the flight fits ask for 0.15-0.25 after the
#:                       bench recalibration and up to 0.8 before it. The range
#:                       spans bench-narrow to flight-wide because the gap is
#:                       physics — a bench motor has no inflow turbulence — and
#:                       which end a real flight sits at is regime dependent.
#: ``phase_diffusion``   the incoherent per-harmonic term. Its share of the
#:                       total width is NOT settled: the estimate that put it
#:                       at 80% came from a demodulation band sized by the old
#:                       inflated width law and is withdrawn. Sampling the
#:                       ratio is the honest response to not knowing it.
#: ``harm_gp_*``         the measured amplitude process: one OU component,
#:                       std 2.90 dB, tau 0.82 s, cross-order coherence 0.191,
#:                       selected by leave-one-rotor-out CV with the
#:                       one-standard-error rule.
#: ``floor_gp_*``        the floor's slow common modulation, fitted 1-2 dB.
#: ``umod_*``            DREGON's per-microphone diaphragm flow noise below
#:                       ~500 Hz, sigma ~3.3 dB, uncorrelated across the array.
#:                       Sampled from zero so the family also contains rigs
#:                       without it (Michael's).
DIVERSE_DYNAMICS: dict[str, Any] = {
    "gamma0_hz": [0.3, 3.0],
    "gamma_slope_hz": [0.02, 0.50],
    "shaft_jitter_rps": [0.02, 0.50],
    "shaft_jitter_tau_s": [0.10, 0.60],
    "shaft_jitter_log_std": [0.20, 0.70],
    "phase_diffusion_hz_per_order": [0.0, 0.06],
    "harm_gp_kernel": "ou",
    "harm_gp_std_db": [1.5, 5.0],
    "harm_gp_tau_s": [0.4, 2.0],
    "harm_coherence": [0.05, 0.45],
    "floor_gp_std_db": [0.5, 3.0],
    "floor_gp_tau_s": [0.5, 4.0],
    "floor_tilt_gp_std": [0.0, 1.0],
    "umod_std_db": [0.0, 3.5],
    "umod_tau_s": [0.10, 0.50],
    "umod_corner_hz": [400.0, 700.0],
}


# ── containment ──────────────────────────────────────────────────────────────
#
# A diverse family that does not CONTAIN the fitted rigs tests the wrong thing.
# If a range chosen by hand for variety happens to exclude the measured value,
# the run answers "does a model trained on a family that excludes reality
# transfer to reality", whose answer is known and uninteresting. So the ranges
# below are not final: every one is widened, programmatically, until it holds
# both fitted rigs' own ranges with margin. Measured before the widening, the
# hand ranges missed on eleven axes — Michael's 0.003-0.008 s shaft correlation
# time against a [0.1, 0.6] prior, both rigs' ~0 static floor against [0.05,
# 0.4], both rigs' 0.4-0.9 Hz/order width against [0.02, 0.50], and more.
#
# Widening cannot fix the PROFILE. The fitted arms carry a measured 200-order
# mean plus a basis; the parametric family here is a power law with blade
# emphasis and independent per-order jitter, and no setting of it produces the
# correlated order-to-order structure a real comb has. The two neighbourhood
# arms below carry the fitted profile populations themselves, with their
# residual spread inflated and the wide dynamics attached — so the measured
# timbres are in support, surrounded rather than merely included.


def _span(values: Any) -> tuple[float, float] | None:
    """``(min, max)`` over a scalar, a range or a fitted per-rotor vector."""
    if values is None:
        return None
    arr = np.atleast_1d(np.asarray(values, dtype=np.float64)).ravel()
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    return float(arr.min()), float(arr.max())


#: Fitted keys that carry the same quantity under a different name, because a
#: fitted preset pins per-rotor vectors where the family samples a range.
_ALIASES: dict[str, tuple[str, ...]] = {
    "gamma0_hz": ("fixed_gamma0_hz",),
    "gamma_slope_hz": ("fixed_gamma_slope_hz",),
    "shaft_jitter_rps": ("fixed_shaft_jitter_rps",),
}


def widen(key: str, base: Any, fitted: list[dict[str, Any]], *, grow: float = 0.2) -> Any:
    """``base`` extended to contain every fitted value of ``key``, plus margin.

    The margin is applied only where a FIT is the binding constraint: a range
    already holding both rigs is returned untouched, and one that clips them is
    opened past them, because a family whose edge sits exactly on the measured
    value puts half of that neighbourhood outside its own support. Padding a
    bound the hand range already cleared would walk ranges past their physical
    meaning — ``floor_rel_db`` into positive numbers, where the floor sits over
    the comb.

    An identically zero range is never widened. C1's zeros ARE C1.
    """
    if not isinstance(base, (list, tuple)) or len(base) != 2:
        return base
    lo, hi = float(base[0]), float(base[1])
    if lo == 0.0 and hi == 0.0:
        return base
    fit_lo, fit_hi = None, None
    for arm in fitted:
        ranges = arm.get("ranges") or {}
        for name in (key, *_ALIASES.get(key, ())):
            span = _span(ranges.get(name))
            if span is None:
                continue
            fit_lo = span[0] if fit_lo is None else min(fit_lo, span[0])
            fit_hi = span[1] if fit_hi is None else max(fit_hi, span[1])
    if fit_lo is None or fit_hi is None:
        return base
    pad = grow * max(fit_hi - fit_lo, abs(hi - lo), 1e-9)
    if fit_lo < lo:
        lo = fit_lo - pad if fit_lo - pad > 0.0 or fit_lo < 0.0 else 0.0
    if fit_hi > hi:
        hi = fit_hi + pad
    return [round(lo, 6), round(hi, 6)]


def contained(policy: dict[str, Any], fitted: dict[str, Any]) -> dict[str, bool]:
    """Per key, whether the control arm's range holds both fitted rigs."""
    arms = [a for a in fitted["sources"]["noise"] if a.get("kind") == "stochastic"]
    wide = policy["sources"]["noise"][0].get("ranges") or {}
    out: dict[str, bool] = {}
    for key, value in wide.items():
        span = _span(value) if isinstance(value, (list, tuple)) and len(value) == 2 else None
        if span is None:
            continue
        ok = True
        for arm in arms:
            ranges = arm.get("ranges") or {}
            for name in (key, *_ALIASES.get(key, ())):
                fit_span = _span(ranges.get(name))
                if fit_span is None:
                    continue
                ok = ok and span[0] <= fit_span[0] and fit_span[1] <= span[1]
        out[key] = ok
    return out


def neighbourhood_arm(
    fitted_arm: dict[str, Any],
    *,
    weight: float,
    dynamics: dict[str, Any],
    inflate_residual_db: float = 3.0,
    rotor_contrast_std_db: float = 3.0,
) -> dict:
    """One fitted rig's measured profile population, widened, with C2 dynamics.

    The profile mean, basis and per-rotor deviations stay: they are the measured
    timbre and the parametric family cannot reproduce them. What widens is the
    spread around them — the per-order residual is inflated so draws sit in a
    NEIGHBOURHOOD of the fit rather than on it, the rotor contrast is opened so
    a clip's rotors can differ more than that rig's did, and the dynamics come
    from the wide family.
    """
    arm = _chassis(fitted_arm)
    arm["weight"] = weight
    ranges = copy.deepcopy(fitted_arm.get("ranges") or {})
    # The residual must cover every harmonic the fitted profile carries, so its
    # length follows the profile mean rather than whatever the fit exported.
    n_harm = len(np.atleast_1d(np.asarray(ranges.get("profile_mean_db") or [0.0])).ravel())
    residual = ranges.get("profile_residual_std_db")
    base = (
        np.zeros(n_harm)
        if residual is None
        else np.resize(np.atleast_1d(np.asarray(residual, dtype=np.float64)).ravel(), n_harm)
    )
    ranges["profile_residual_std_db"] = [
        float(v) for v in np.hypot(base, float(inflate_residual_db))
    ]
    ranges["rotor_contrast_std_db"] = float(
        max(float(ranges.get("rotor_contrast_std_db") or 0.0), rotor_contrast_std_db)
    )
    ranges.update(dynamics)
    ranges.update(LABEL_ERROR)
    ranges["min_lines_above_floor"] = 0.30
    ranges["min_lines_above_floor_per_rotor"] = 0.20
    # The per-microphone vectors are dropped: microphone INDEX carries no
    # measured physics, and pinning them is what made channel 0 of DREGON score
    # twice the error of the full array.
    for key in ("fixed_mic_gain_db", "fixed_mic_floor_db", "fixed_mic_gain_all_db"):
        ranges.pop(key, None)
    ranges.setdefault("mic_gain_all_db", [0.0, 6.0])
    ranges.setdefault("mic_floor_std_db", [0.0, 3.0])
    arm["ranges"] = ranges
    arm["fitted_from"] = fitted_arm.get("fitted_from")
    return arm


def _chassis(arm: dict[str, Any]) -> dict[str, Any]:
    """One fitted arm stripped to what is not a claim about a rig."""
    out = {k: copy.deepcopy(v) for k, v in arm.items() if k not in ("ranges", "fitted_from")}
    out.pop("weight", None)
    return out


def wide_arm(
    fitted_arm: dict[str, Any],
    *,
    weight: float,
    dynamics: dict[str, Any],
    fitted: list[dict[str, Any]] | None = None,
) -> dict:
    """A control arm: the fitted chassis, the wide timbre family, and either
    the static or the diverse dynamics.

    ``fitted`` are the arms whose measured ranges the family must contain; the
    timbre and floor ranges are widened to hold them. The DYNAMICS are widened
    only when they are meant to be present — C1's zeros are the point of C1 and
    are left alone.
    """
    arm = _chassis(fitted_arm)
    arm["weight"] = weight
    ranges = dict(PROFILE_FAMILY | LABEL_ERROR | dynamics)
    if fitted:
        # Only the TIMBRE and floor ranges are stretched to the fits. Stretching
        # a uniform to reach one rig's widest fitted rotor would put half the
        # family's mass wider than any measurement: widening shaft_jitter_rps to
        # 1.86 rev/s gave a median half width of 20 Hz at order 16 against an
        # 80 Hz line spacing, and 16 draws in 150 came out as a merged
        # continuum — the exact failure of the original stochastic stream. The
        # fitted extremes live in the neighbourhood arms, which pin them
        # directly, so the MIXTURE contains them without the wide arm becoming
        # a harder task than reality.
        ranges = {
            key: (widen(key, value, fitted) if key in PROFILE_FAMILY else value)
            for key, value in ranges.items()
        }
    arm["ranges"] = ranges
    return arm


def matched_static_arm(fitted_arm: dict[str, Any], *, weight: float) -> dict:
    """One fitted rig's own timbre with the dynamics switched off.

    This is what makes C1 a fair control rather than a strawman: the family
    demonstrably contains the measured profiles, so if C1 fails it cannot be
    because the right timbre was never drawn.
    """
    arm = _chassis(fitted_arm)
    arm["weight"] = weight
    ranges = copy.deepcopy(fitted_arm.get("ranges") or {})
    # A fitted preset pins per-rotor vectors that OVERRIDE the sampled ranges,
    # so zeroing `shaft_jitter_rps` while `fixed_shaft_jitter_rps` survives
    # leaves the lines exactly as wide as the fit made them. Measured on the
    # first generated file: half width 15 Hz at order 16, on an 80 Hz spacing,
    # in an arm whose whole purpose is to have no line broadening at all.
    for key in ("fixed_shaft_jitter_rps", "fixed_gamma0_hz", "fixed_gamma_slope_hz"):
        ranges.pop(key, None)
    ranges.update(STATIC_OFF)
    ranges.update(LABEL_ERROR)
    ranges["min_lines_above_floor_per_rotor"] = 0.20
    arm["ranges"] = ranges
    arm["fitted_from"] = fitted_arm.get("fitted_from")
    return arm


def build_static_policy(fitted: dict[str, Any]) -> dict[str, Any]:
    """C1 — static combs, wide timbre, plus the two fitted timbres static."""
    noise = fitted["sources"]["noise"]
    stochastic = [a for a in noise if a.get("kind") == "stochastic"]
    other = [a for a in noise if a.get("kind") != "stochastic"]
    if len(stochastic) != 2:
        raise ValueError(f"expected two fitted rig arms, got {len(stochastic)}")
    out = copy.deepcopy(fitted)
    out["sources"]["noise"] = [
        wide_arm(stochastic[0], weight=0.5, dynamics=STATIC_OFF, fitted=stochastic),
        matched_static_arm(stochastic[0], weight=0.15),
        matched_static_arm(stochastic[1], weight=0.15),
        *copy.deepcopy(other),
    ]
    return out


def build_diverse_policy(fitted: dict[str, Any]) -> dict[str, Any]:
    """C2 — one wide arm of the full model, measured dynamics, wide timbre."""
    noise = fitted["sources"]["noise"]
    stochastic = [a for a in noise if a.get("kind") == "stochastic"]
    other = [a for a in noise if a.get("kind") != "stochastic"]
    if not stochastic:
        raise ValueError("fitted policy has no stochastic arm to take a chassis from")
    arm = wide_arm(stochastic[0], weight=0.4, dynamics=DIVERSE_DYNAMICS, fitted=stochastic)
    # The speed laws are measured on the bench, not inherited: line power grows
    # as s^2.196 +- 0.125 over 802 cells and the floor as s^1.400 +- 0.076, so
    # the comb's PROMINENCE grows as s^0.80 instead of being speed-invariant by
    # construction as a single shared exponent makes it.
    arm["amp_rps_exponent"] = 2.196
    arm["amp_rps_exponent_floor"] = 1.400
    dynamics = {key: widen(key, value, stochastic) for key, value in DIVERSE_DYNAMICS.items()}
    out = copy.deepcopy(fitted)
    out["sources"]["noise"] = [
        arm,
        # The measured timbres, surrounded: the parametric family above cannot
        # reach a 200-order measured profile, so each fitted population enters
        # with its residual spread inflated and the wide dynamics attached.
        neighbourhood_arm(stochastic[0], weight=0.2, dynamics=dynamics),
        neighbourhood_arm(stochastic[1], weight=0.2, dynamics=dynamics),
        *copy.deepcopy(other),
    ]
    return out
