"""A whole-flight generative rotor-noise model assembled from PRE-REVISION fits.

The stage-2 ("previous model") fits are per-OPERATING-POINT: one parameter
vector per regime, fitted on a window where the rotors barely move. A flight is
not one operating point — it starts at rest, spools up, hovers, cruises — so a
single fit cannot render it. This module assembles the fits it has into one
speed-indexed generator, with exact silence at zero speed.

The assembly
------------
Each fit becomes an :class:`Anchor` tagged with the MEAN ROTOR SPEED of its own
fitted support. A zero-speed anchor is implicit and is SILENCE (no export). On
the anchor speed grid ``[0, s_1, ..., s_A]`` the model puts piecewise-linear hat
weights (:func:`blend_weights`), clamped above the top anchor, and renders as::

    y(t) = sum_a w_a(s(t)) * x_a(t)

where ``s(t)`` is the smoothed mean rotor speed and ``x_a`` is anchor ``a``
rendered over the WHOLE trajectory by
:func:`experiments.stochastic_fit.stage2.render_from_export`. Every anchor is
rendered with the SAME seed, the same microphone count, the same work rate and
the same ``(n_rotors, K)`` order grid, so ``params_from_export`` derives the
same ``n_harmonics`` and every anchor's render consumes the SAME sequence of
random draws: the floor noise, the per-(mic, order) propagation phases and the
shaft phase are bit-identical across anchors. The anchor renders are therefore
PHASE-LOCKED and the time-domain blend acts as an amplitude interpolation of
the same comb rather than as a sum of two independent noises.

What that interpolation is, exactly
-----------------------------------
Honest accounting of the approximation:

* EXACT parameter interpolation for anything that enters the render linearly in
  amplitude at fixed phase — the per-(rotor, order) line amplitudes
  ``profile_db`` and the fixed per-(mic, rotor) line gains ``mic_gain_db``. A
  weighted sum of two phase-locked tone banks IS the tone bank of the weighted
  sum of their amplitudes.
* An APPROXIMATION for every parameter that shapes the spectrum non-linearly:
  the line widths (``gamma0``, ``gamma_slope``), the floor shape
  (``floor_shape_db``, ``floor_tilt_db_oct``), the coherent/incoherent split
  (``coherence_k_half``) and the speed exponents (``amp_exp``, ``floor_exp``).
  Mid-blend the render carries a MIXTURE of two widths and two floor shapes,
  not the interpolated width and shape. The floor parts of two anchors are
  driven by the same white noise, so they add in amplitude rather than in
  power; the narrowband-noise share of the comb does likewise.
* NOT a statement about physics between the anchors. Nothing here was fitted
  between the operating points; the blend is an interpolation rule chosen for
  continuity and exact silence, and it is only as good as the two endpoints.

Levels
------
A stage-2 fit normalises its periodogram to unit mean and exports the
parameters in THAT unit (``scores.power_scale``), so two clips' levels are not
comparable until the scale is folded back. Anchors are therefore built through
:func:`physical_export`, which reuses the repo's declared path —
:func:`revised_eval._to_physical` for the per-clip fold and
:func:`revised_eval.to_renderer_units` for the sample-rate term — and
:func:`assemble` REFUSES an anchor that has not. Rendering with
``normalize_rms=None`` then keeps that physical level, so a level RAMP across
a spool-up and the level STEP between regimes are predictions of the model
rather than artefacts of per-window normalisation.

Zero-speed silence
------------------
Silence comes from the WEIGHTS, not from the anchors. A stage-2 export keeps a
speed-independent ``floor_static_rel`` term (the recording chain's own floor),
so an anchor rendered at zero speed is not silent; the hat weight of every
anchor is exactly ``0.0`` at ``s = 0``, so the blend is exactly zero there.
This holds only if the anchor renders are FINITE, which is why
:func:`assemble` rebases a fitted floor exponent that is negative — see
:func:`_rebase_divergent_floor`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from experiments.stochastic_fit import stage2 as S2

#: Level given to orders an anchor did not fit, when a shorter anchor profile is
#: padded onto the common order grid. 200 dB below the fitted comb is silence in
#: float64 and keeps the padded orders out of every spectrum.
PROFILE_PAD_DB = -200.0
#: Per-order arrays of a stage-2 export and how a padded order continues.
#: ``profile_db`` (R, K) is a level, ``h_db`` (R, K, J) a mean-zero GP deviation,
#: ``gamma`` (R, K) a line width whose only sane continuation is the last
#: fitted order's.
_PER_ORDER: dict[str, str] = {"profile_db": "pad", "h_db": "zero", "gamma": "edge"}
#: ``stochastic_rotor_noise.build_psd``'s reference speed for the level laws.
AMP_RPS_REF = 80.0
#: The stage-2 render grid these anchors are rendered on. The level conversion
#: MUST be taken at the grid the render actually uses, not at the declared
#: 64 kHz one: :func:`revised_eval.rate_factor_db` is a function of the work
#: rate, and 44.1 kHz is ``stage2.render_from_export``'s own default.
SAMPLE_RATE_WORK = 44100
#: Keys the physical-level path writes into an export. :func:`assemble` refuses
#: an anchor without them, so the level path cannot be skipped by accident —
#: which is exactly the defect this model shipped with first: it fed
#: ``render_from_export`` the RAW per-clip ``params``, whose levels are in each
#: clip's own ``scores.power_scale`` unit, and two anchors therefore carried
#: two incompatible scales.
LEVEL_MARKERS = ("power_scale_folded_db", "render_units_rate_factor_db")


@dataclass(frozen=True)
class Anchor:
    """One fitted operating point of the flight model.

    Attributes:
        speed_rps: the mean rotor speed of the fit's OWN support, measured from
            telemetry — the speed at which this anchor is the whole model.
        export: the per-clip stage-2 parameter dict in PHYSICAL renderer units,
            as :func:`physical_export` produces it and
            :func:`stage2.render_from_export` consumes it.
        label: short name used in figures and diagnostics.
        notes: transformations :func:`assemble` applied to ``export``, recorded
            so a render can report them instead of hiding them.
    """

    speed_rps: float
    export: dict[str, Any]
    label: str
    notes: tuple[str, ...] = field(default=())


def physical_export(
    entry: Mapping[str, Any], *, sample_rate_work: int = SAMPLE_RATE_WORK
) -> dict[str, Any]:
    """One clip's stage-2 export in PHYSICAL renderer units.

    Two declared steps, neither of them reimplemented here:

    1. :func:`revised_eval._to_physical` folds the clip's own
       ``scores.power_scale`` into ``revised_eval.ABSOLUTE_POWER_DB_FIELDS``
       (``profile_db``, ``floor_mean_db``). A fit normalises its periodogram to
       unit mean and exports the parameters in THAT scaled unit, so two clips'
       levels are not comparable until the scale is folded back. For these
       anchors the folded offsets are about -39.6 dB (FLY125 standby), -24.1 dB
       (FLY125 cruise) and -22.3 dB (DREGON room-2 cruise): 15.5 dB between the
       two Michael anchors alone.
    2. :func:`revised_eval.to_renderer_units` applies the declared sample-rate
       term ``10 log10(work / analysis)`` to the same two fields. It is taken at
       ``sample_rate_work``, the grid this model actually renders on, so the
       historical 44.1 kHz path contributes +4.403 dB rather than the declared
       64 kHz path's +6.021 dB. This term is common to every anchor, so it sets
       the absolute level and cannot change the level STEP between anchors.

    Valid ONLY with ``normalize_rms=None``, which is what
    :func:`render_flight` passes: ``synthesize``'s default rescales the whole
    waveform and erases exactly the quantity this function establishes.

    The result is the source-side level of ONE declared observation chain
    (:data:`stage2.RENDER_TRANSFER_SPEC`), not a calibrated absolute source
    spectrum — see :func:`revised_eval.to_renderer_units`' own docstring.

    Args:
        entry: the per-clip fit ENTRY, i.e. ``{"params": ..., "scores": ...}``.
            The raw ``params`` dict alone is not enough: ``power_scale`` lives
            under ``scores``.
    """
    from experiments.stochastic_fit import revised_eval as RE

    params = RE._to_physical(entry)
    mp = RE.to_renderer_units(
        RE.ModelParams(params=params, spec={}, source={}),
        sample_rate_work=int(sample_rate_work),
    )
    out = dict(mp.params)
    out["render_units_rate_factor_db"] = float(mp.source["render_units"]["rate_factor_db"])
    out["render_units_sample_rate_work"] = int(sample_rate_work)
    return out


def anchor_from_entry(
    entry: Mapping[str, Any],
    *,
    speed_rps: float,
    label: str,
    sample_rate_work: int = SAMPLE_RATE_WORK,
) -> Anchor:
    """An :class:`Anchor` built through the physical-level path.

    The only supported way to make an anchor from a fit file: it routes the
    export through :func:`physical_export`, so the levels of two anchors are
    on one scale before they are ever blended.
    """
    export = physical_export(entry, sample_rate_work=sample_rate_work)
    return Anchor(
        speed_rps=float(speed_rps),
        export=export,
        label=label,
        notes=(
            f"power_scale folded {export['power_scale_folded_db']:+.3f} dB; "
            f"render units {export['render_units_rate_factor_db']:+.3f} dB at "
            f"{sample_rate_work} Hz work grid",
        ),
    )


@dataclass(frozen=True)
class FlightModel:
    """Anchors on a common order grid, sorted by speed, plus implicit silence."""

    rig: str
    anchors: tuple[Anchor, ...]

    @property
    def speeds(self) -> np.ndarray:
        """``(A,)`` anchor speeds in rev/s, strictly increasing."""
        return np.array([a.speed_rps for a in self.anchors], dtype=np.float64)

    @property
    def n_orders(self) -> int:
        """``K``, the shared number of harmonic orders."""
        return int(np.asarray(self.anchors[0].export["profile_db"]).shape[1])

    @property
    def n_rotors(self) -> int:
        return int(np.asarray(self.anchors[0].export["profile_db"]).shape[0])

    @property
    def region_labels(self) -> tuple[str, ...]:
        """Names of the ``A + 1`` blend regions of the speed grid."""
        names = ["silence"] + [a.label for a in self.anchors]
        regions = [f"{names[i]}->{names[i + 1]}" for i in range(len(self.anchors))]
        return tuple([*regions, f">={self.anchors[-1].label}"])


@dataclass(frozen=True)
class FlightRender:
    """One blended flight render and what it took."""

    audio: np.ndarray
    speed_rps: np.ndarray
    weights: np.ndarray
    diagnostics: dict[str, Any]


def _regrid(export: dict[str, Any], k: int) -> dict[str, Any]:
    """A copy of ``export`` whose per-order arrays all live on ``K = k``."""
    out = dict(export)
    for name, how in _PER_ORDER.items():
        if name not in out:
            continue
        a = np.asarray(out[name], dtype=np.float64)
        if a.shape[1] == k:
            out[name] = a
            continue
        if a.shape[1] > k:
            out[name] = a[:, :k].copy()
            continue
        pad = [(0, 0)] * a.ndim
        pad[1] = (0, k - a.shape[1])
        if how == "edge":
            out[name] = np.pad(a, pad, mode="edge")
        else:
            out[name] = np.pad(
                a, pad, mode="constant", constant_values=PROFILE_PAD_DB if how == "pad" else 0.0
            )
    return out


def _rebase_divergent_floor(export: dict[str, Any], speed_rps: float) -> tuple[dict[str, Any], str]:
    """Make an anchor's broadband floor FINITE at zero speed, level-preserving.

    ``build_psd`` scales the rotors' share of the floor by
    ``(s / 80)**floor_exp``. A fit taken at a single operating point does not
    identify that exponent, and the pre-revision DREGON room-2 fit came out
    with ``floor_exp = -3.71``: its floor DIVERGES as the rotors stop, so at a
    telemetry sample of exactly zero rev/s the render is ``inf`` and the
    overlap-add smears the non-finite value over a whole synthesis window.

    The fix is the smallest one that keeps the fit's own statement where the
    fit has support: set ``floor_exp = 0`` and fold the anchor's own floor gain
    AT ITS OWN SPEED into ``floor_mean_db``, so the floor is unchanged at
    ``speed_rps`` and simply stops extrapolating below it. The model's
    low-speed floor is then carried by the blend weight alone. Applied ONLY
    when the fitted exponent is negative; a non-negative exponent already
    vanishes at zero speed and is left exactly as fitted.
    """
    floor_exp = float(export.get("floor_exp", export.get("amp_exp", 0.0)))
    if floor_exp >= 0.0:
        return export, ""
    rel = max(float(export.get("floor_static_rel", 0.0)), 0.0)
    gain_at_anchor = (max(speed_rps, 1e-9) / AMP_RPS_REF) ** floor_exp + rel
    shift_db = 10.0 * np.log10(gain_at_anchor / (1.0 + rel))
    out = dict(export)
    out["floor_exp"] = 0.0
    out["floor_mean_db"] = float(export.get("floor_mean_db", 0.0)) + float(shift_db)
    return out, (
        f"floor_exp {floor_exp:+.3f} diverges at zero speed; rebased to 0.0 with "
        f"floor_mean_db {shift_db:+.3f} dB so the floor is unchanged at "
        f"{speed_rps:.2f} rev/s and constant below it"
    )


def assemble(rig: str, anchors: Sequence[Anchor]) -> FlightModel:
    """Validate anchors, put them on one order grid, return the flight model.

    The common grid is ``K = max`` over the anchors, with shorter profiles
    padded at :data:`PROFILE_PAD_DB`. Padding rather than trimming to the
    minimum is deliberate: the FLY125 standby fit reaches order 230 (7900 Hz at
    35 rev/s) against the cruise fit's 111, and 15.6% of its fitted line power
    sits above order 111 — trimming would delete the standby comb between 4 and
    7.9 kHz, which is exactly the band a spool-up render is judged on. The
    padded orders are inaudible for the shorter anchor and, at ITS speed, above
    the output Nyquist anyway.

    Every anchor MUST have come through :func:`physical_export` — checked, not
    assumed. Blending exports that are each in their own ``power_scale`` unit
    is silently wrong rather than loudly wrong, so the check is here.
    """
    if not anchors:
        raise ValueError(f"{rig}: a flight model needs at least one anchor")
    unscaled = [a.label for a in anchors if any(m not in a.export for m in LEVEL_MARKERS)]
    if unscaled:
        raise ValueError(
            f"{rig}: anchor(s) {unscaled} are not in physical renderer units (missing "
            f"{LEVEL_MARKERS}); build them with flight_model.anchor_from_entry / "
            "physical_export, never from a raw per-clip params dict"
        )
    speeds = [float(a.speed_rps) for a in anchors]
    if any(s <= 0.0 for s in speeds):
        raise ValueError(
            f"{rig}: anchor speeds must be > 0 (zero speed is the implicit silence "
            f"anchor), got {speeds}"
        )
    if any(b <= a for a, b in zip(speeds, speeds[1:], strict=False)):
        raise ValueError(f"{rig}: anchor speeds must be strictly increasing, got {speeds}")
    profiles = [
        np.atleast_2d(np.asarray(a.export["profile_db"], dtype=np.float64)) for a in anchors
    ]
    rotors = {p.shape[0] for p in profiles}
    if len(rotors) != 1:
        raise ValueError(
            f"{rig}: anchors disagree on n_rotors: "
            + ", ".join(f"{a.label}={p.shape[0]}" for a, p in zip(anchors, profiles, strict=True))
        )
    k = max(p.shape[1] for p in profiles)
    out: list[Anchor] = []
    for a in anchors:
        export, note = _rebase_divergent_floor(a.export, float(a.speed_rps))
        notes = (*a.notes, note) if note else a.notes
        out.append(replace(a, export=_regrid(export, k), notes=notes))
    return FlightModel(rig=rig, anchors=tuple(out))


def blend_weights(model: FlightModel, speed_rps: np.ndarray) -> np.ndarray:
    """``(A, T)`` piecewise-linear hat weights over ``[0] + anchor speeds``.

    Anchor ``a`` owns the hat that is 1 at its own speed and falls linearly to
    0 at its neighbours' speeds; the lowest anchor's left neighbour is the
    implicit zero-speed SILENCE anchor, whose row is not returned because it
    contributes nothing. Above the top anchor the weights are constant (the
    model does not extrapolate past its fastest fit). The rows therefore sum to
    1 everywhere except below the lowest anchor speed, where they sum to
    ``s / s_1`` — the missing share is silence — and to exactly 0 at ``s = 0``.
    """
    s = np.asarray(speed_rps, dtype=np.float64).reshape(-1)
    grid = np.concatenate(([0.0], model.speeds))
    w = np.empty((len(model.anchors), s.size), dtype=np.float64)
    for i in range(len(model.anchors)):
        hat = np.zeros(grid.size, dtype=np.float64)
        hat[i + 1] = 1.0
        w[i] = np.interp(np.maximum(s, 0.0), grid, hat, left=0.0, right=hat[-1])
    return w


def smooth_speed(
    rps: np.ndarray, *, sample_rate: int = S2.SR, smooth_s: float = 0.05
) -> np.ndarray:
    """``(T,)`` mean-over-rotors speed, centred moving average of ``smooth_s``.

    Edge-padded, so a trajectory that STARTS at rest still reads as rest at
    ``t = 0`` instead of being pulled towards the first fast sample. An
    all-zero trajectory smooths to exact zeros, which is what makes the
    silence check exact.
    """
    s = np.atleast_2d(np.asarray(rps, dtype=np.float64)).mean(axis=0)
    n = max(int(round(smooth_s * sample_rate)), 1)
    if n <= 1:
        return s
    lo = n // 2
    padded = np.pad(s, (lo, n - 1 - lo), mode="edge")
    c = np.concatenate(([0.0], np.cumsum(padded)))
    return (c[n:] - c[:-n]) / float(n)


def render_flight(
    model: FlightModel,
    rps: np.ndarray,
    *,
    sample_rate_work: int = 44100,
    n_mics: int = 8,
    seed: int = 0,
    smooth_s: float = 0.05,
) -> FlightRender:
    """Render ``model`` over one rotor-speed trajectory and blend the anchors.

    Args:
        model: the assembled model; its anchors share one order grid, which is
            what makes their random draws identical.
        rps: ``(R, T)`` rotor speeds in rev/s at 16 kHz — RAW telemetry, since
            the blend is indexed by the speed a controller actually commanded.
        sample_rate_work: the render grid handed to
            :func:`stage2.render_from_export`; 16 kHz output either way.
        n_mics: channels; all anchors get the same count, so their fixed mic
            gains and propagation phases line up.
        seed: shared by every anchor render. Not optional in spirit: two
            anchors rendered under different seeds would add as independent
            noises and the blend would no longer be an amplitude interpolation.
        smooth_s: the moving-average window of the speed that indexes the
            blend. 50 ms is short against any spool-up and long enough to keep
            telemetry quantisation out of the weights.

    Returns:
        A :class:`FlightRender`. ``audio`` is ``(M, T)`` float64 at 16 kHz at
        the MODEL'S OWN absolute level (every anchor is rendered with
        ``normalize_rms=None``), so it is directly comparable to a real clip
        only after ONE declared scalar.
    """
    rps = np.atleast_2d(np.asarray(rps, dtype=np.float64))
    renders = [
        S2.render_from_export(
            a.export,
            rps,
            sample_rate_work=sample_rate_work,
            n_mics=n_mics,
            seed=seed,
            normalize_rms=None,
        )
        for a in model.anchors
    ]
    lengths = sorted({int(r.shape[-1]) for r in renders} | {int(rps.shape[-1])})
    n = lengths[0]
    renders = [r[:, :n] for r in renders]
    bad = [a.label for a, r in zip(model.anchors, renders, strict=True) if not np.isfinite(r).all()]
    if bad:
        raise ValueError(
            f"{model.rig}: anchor render(s) {bad} are not finite; a fitted level law "
            "diverges on this trajectory and the blend cannot be taken"
        )

    speed = smooth_speed(rps, smooth_s=smooth_s)[:n]
    weights = blend_weights(model, speed)
    audio = np.zeros_like(renders[0])
    for w, r in zip(weights, renders, strict=True):
        audio += w[None, :] * r

    grid = np.concatenate(([0.0], model.speeds))
    region = np.clip(np.searchsorted(grid, speed, side="right") - 1, 0, grid.size - 1)
    diagnostics: dict[str, Any] = dict(
        rig=model.rig,
        anchor_labels=[a.label for a in model.anchors],
        anchor_speeds_rps=[round(float(a.speed_rps), 4) for a in model.anchors],
        anchor_notes={a.label: list(a.notes) for a in model.anchors},
        anchor_level_db={
            a.label: {
                "power_scale_folded_db": round(float(a.export["power_scale_folded_db"]), 4),
                "render_units_rate_factor_db": round(
                    float(a.export["render_units_rate_factor_db"]), 4
                ),
            }
            for a in model.anchors
        },
        n_orders=model.n_orders,
        n_rotors=model.n_rotors,
        n_mics=int(n_mics),
        seed=int(seed),
        sample_rate_work=int(sample_rate_work),
        smooth_s=float(smooth_s),
        length_samples=int(n),
        length_trim_samples=[int(x - n) for x in lengths],
        anchor_rms={
            a.label: round(float(np.sqrt(np.mean(np.square(r)))), 8)
            for a, r in zip(model.anchors, renders, strict=True)
        },
        anchor_weight_mean={
            a.label: round(float(w.mean()), 5) for a, w in zip(model.anchors, weights, strict=True)
        },
        region_fraction={
            name: round(float(np.mean(region == i)), 5)
            for i, name in enumerate(model.region_labels)
        },
        speed_rps_range=[round(float(speed.min()), 4), round(float(speed.max()), 4)],
    )
    return FlightRender(audio=audio, speed_rps=speed, weights=weights, diagnostics=diagnostics)
