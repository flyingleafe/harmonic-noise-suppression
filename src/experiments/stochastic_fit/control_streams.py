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


def _chassis(arm: dict[str, Any]) -> dict[str, Any]:
    """One fitted arm stripped to what is not a claim about a rig."""
    out = {k: copy.deepcopy(v) for k, v in arm.items() if k not in ("ranges", "fitted_from")}
    out.pop("weight", None)
    return out


def wide_arm(fitted_arm: dict[str, Any], *, weight: float, dynamics: dict[str, Any]) -> dict:
    """A control arm: the fitted chassis, the wide timbre family, and either
    the static or the diverse dynamics."""
    arm = _chassis(fitted_arm)
    arm["weight"] = weight
    arm["ranges"] = PROFILE_FAMILY | LABEL_ERROR | dynamics
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
        wide_arm(stochastic[0], weight=0.5, dynamics=STATIC_OFF),
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
    arm = wide_arm(stochastic[0], weight=0.8, dynamics=DIVERSE_DYNAMICS)
    # The speed laws are measured on the bench, not inherited: line power grows
    # as s^2.196 +- 0.125 over 802 cells and the floor as s^1.400 +- 0.076, so
    # the comb's PROMINENCE grows as s^0.80 instead of being speed-invariant by
    # construction as a single shared exponent makes it.
    arm["amp_rps_exponent"] = 2.196
    arm["amp_rps_exponent_floor"] = 1.400
    out = copy.deepcopy(fitted)
    out["sources"]["noise"] = [arm, *copy.deepcopy(other)]
    return out
