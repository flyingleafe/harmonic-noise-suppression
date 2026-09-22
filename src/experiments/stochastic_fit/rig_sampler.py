"""Parameter-neighbourhood sampler for pre-revision stage-2 rig exports.

What this is
------------
Given fitted stage-2 rig exports (the coherent+Lorentzian family whose
per-clip parameter dict is loaded the way ``scripts/_revised_ab_render.py``'s
``prev_export`` loads it), this module draws perturbed exports that are
plausible *other drones of the same kind*, or other operating points of the
same drone. Every result is a new export of the same schema, renderable by
:func:`experiments.stochastic_fit.stage2.render_from_export` unchanged. There
are two modes:

* :func:`sample_rig` — a close NEIGHBOURHOOD of ONE fit, at width
  ``strength``.
* :func:`sample_path` — a CLOUD along the path between TWO fits: interpolate
  the decomposed coordinates at a mixing coordinate ``t`` (uniform when not
  given), then apply the same neighbourhood draw at width ``spread`` around
  that point. With ``t`` uniform, a Michael anchor and a DREGON anchor give a
  cloud in which both rigs and everything between them appear, and none is
  especially likely.

The structural invariants of a rotor comb — the even/odd blade-passing parity
split keeping its sign, and the harmonic profile falling with order like a
power law in ``log`` order — are HARD guards in both modes: a draw that breaks
either is rejected and redrawn, never clipped into shape. Same for a finite
render and a non-negative ``floor_exp``. Everything else about the profiles and
the per-rotor gains moves freely; linewidths move only a little.

This is a generator of varied synthetic rigs. It is NOT campaign scoring and it
makes no gate claim: nothing here says a sampled rig would pass any acceptance
probe, only that it satisfies the stated plausibility guards.

Levels: the physical scale, the band, and what is left over
-----------------------------------------------------------
A stage-2 fit exports its levels in the scaled unit of its own clip's
periodogram normalisation, so the raw ``params`` dict is NOT comparable in
absolute level across clips or rigs. Every anchor is loaded through
:func:`load_anchor` -> :func:`to_physical`, which applies BOTH halves of the
declared conversion — ``scores.power_scale`` folded per clip, then
``to_renderer_units``' ``10 log10(work / analysis)`` = +4.4032 dB at the
44.1 kHz work grid these renders use — and phase A does the same for every
clip of every fit before any measurement.

Every "level" in this module is then the power in :data:`LEVEL_BAND_HZ`
(300 Hz - 7.9 kHz), NOT a whole-array RMS. The reason is measured: on the
corrected scale each anchor's render agrees with its OWN real clip to
-1.81 dB (FLY125 cruise) and +0.58 dB (DREGON room-2 cruise) in that band,
with an RMS band deviation of 2.09 and 0.59 dB — these fits are right above
300 Hz. Below it the render carries a large excess concentrated in rotor
orders k=1-3 (the floor BETWEEN the lines is right to about 1 dB), a localised
render-path defect under investigation elsewhere. A whole-array RMS is
power-weighted, so those few orders dominate it: the same two exports differ
by +2.467 dB on the band coordinate and by +15.750 dB on the whole-array RMS.

On the band coordinate the FLY125-to-DREGON level offset decomposes as
``+2.467 = +0.087 (the two REAL recordings) + 0.575 (DREGON's fit error)
+ 1.805 (FLY125's fit error)``. The two real recordings are within 0.09 dB of
each other, so whatever level difference the exports carry is fit error, and
it is now small. An earlier revision of this module reported that same offset
as 12.342 dB with DREGON "+11.16 dB too loud"; that was the confounded
whole-array-RMS coordinate with the rate term omitted, and it is retracted —
see :func:`to_physical` for the band-resolved evidence that settles it.

:func:`sample_path` measures each anchor's own level (:func:`render_level_db`,
the band level of its own render at ``normalize_rms=None`` on its own
telemetry), removes it into ONE scalar coordinate carried by ``gain_all_db``
(:func:`level_free`), interpolates the shape coordinates on that common scale,
and adds the interpolated level back. The removed offset is recorded in the
sample's ``_sampler.path`` block and printed by the demo. Stated honestly:
because every step is linear in dB, the default dB-linear level path
reconstructs exactly what naive interpolation of the level fields would give —
the split is what makes the offset MEASURED and recorded instead of implicit,
and what makes the ``level_db`` override (pin a whole cloud to one playback
level) possible.

What varies, and by how much
----------------------------
Every width is measured in :func:`measure_structure` (phase A) and cached in
``results/rig_sampler/structure.json``; the frozen numbers below are that
file's ``widths_strength1`` block. ``main()`` re-measures and prints frozen
against measured so the two provably cannot drift apart. There are three
measured tiers, and they mean very different things:

* **between-clip** — same rig, same regime, a different 8 s window. This is
  the ``strength = 1`` target. Only THREE quantities actually vary between
  clips, because the stage-2 rig fit TIES everything else across the clips of
  one rig (``rig_spec.tie_profile/tie_width/tie_floor_shape/...``): the
  per-rotor profile gain, ``floor_mean_db`` and ``coherence_k_half``. The
  between-clip spread of every other quantity is identically zero — not small,
  zero — so it cannot be used as a width.
* **between-refit** — same rig, same recordings, a different fit
  configuration (FLY125 cruise base/refined/dyn; DREGON room-2 cruise
  refined/flight). This is the EXPLICITLY-LABELLED FALLBACK used for every
  tied quantity. It is the tightest measured non-zero neighbourhood there is:
  it says how far the parameter can move while still describing the same
  drone on the same audio. Per-draw sigma is taken as
  ``median|delta| / sqrt(2)`` (two independent draws differ with variance
  ``2 sigma^2``), the median rather than the mean because two of the pairs
  cross an identifiability degeneracy (see ``amp_exp`` below).
* **between-rig** — FLY125 cruise vs FLY125 standby vs DREGON. Reported for
  context only; it is enormous (``gamma_slope`` spans 14.7x, ``gamma0`` spans
  0 to 13.1 bins) and is what ``strength`` walks towards, not what it is
  calibrated on.

Strength ladder, measured: at ``strength = 1`` the profile-shape widths equal
the between-refit spread; the between-rotor spread within one rig is reached
at ``strength ~ 3`` (slope 4.40 against 1.46 dB/decade) and the between-rig
spread at ``strength ~ 5`` (slope 6.93). ``strength = 2`` — used by the demo —
is therefore "a noticeably different rotor set on the same airframe".

Profile decomposition
---------------------
Per rotor, over orders ``k = 1..K`` with ``x = log10 k``:

    profile_db[r, k] = gain_r + slope_r * (x - mean x) + env_r(k) * s(k) + resid_r(k)

``s(k) = +1`` on even orders, ``-1`` on odd. Two-bladed rotors put their
blade-passing energy on the even orders and the split is large: 9.8 dB mean
amplitude over ``k <= 16`` on FLY125 cruise, 8.5 dB over all 124 measured
rotor profiles. A SINGLE parity amplitude does not describe it, because the
split DECAYS with order (13.6 dB over k<=16, 3.1 dB over k=33..48, 0.0 dB
above k=80 on FLY125 cruise rotor 0), so ``env_r(k)`` is kept as a measured
envelope — the local 3-point alternating amplitude smoothed over +-0.12
decades — and the sampler scales it rather than replacing it. That choice is
quantified: mean residual std over the 124 measured rotor profiles is
5.08 dB for this decomposition against 6.33 dB for the requested OLS design
``[1, x, s]`` and 5.50 dB for separate even/odd trends ``[1, x, s, s*x]``
(which is itself better than the single trend on 118 of 124 profiles). The
envelope form wins on 124 of 124, so it is what the sampler uses; the
``[1, x, s]`` slopes are also reported because they are the numbers phase A
was asked for and they reproduce the headline slopes exactly
(-9.31/-18.85/-15.52/-6.81 dB/decade on FLY125 cruise).

``resid_r`` is NOT white in ``log k``: its squared-exponential correlation
length is 0.073 decades (variogram estimate, median over 124 rotor profiles;
0.045 decades from the small-lag autocorrelation fit, the same number within
the bias of detrending). Squared-exponential halves the mean restricted-lag
ACF error against exponential (0.0133 against 0.0251), though per profile it
wins only 65 of 124 times. A stationary kernel in
``log k`` is the right parameterisation precisely because it reproduces the
observed non-stationarity in ORDER: 0.073 decades is much shorter than the
0.30-decade gap from k=1 to k=2 (so low orders are independent and jumpy) and
much longer than the 0.011-decade gap at k=40 (so high orders are smooth).
Redrawing the residual with this kernel is what makes a sampled profile look
like a drone instead of like white noise.

Widths at strength 1 (dB unless stated; provenance in
``structure.json:widths_strength1``)

    per-rotor profile gain        3.577      between-clip (RMS of 16 per-(fit,rotor) stds)
    trend slope (dB/decade)       1.456      between-refit, median|d| 2.059
    parity envelope scale (ln)    0.080      between-refit, median|d ln| 0.113
    residual redraw (frac sigma)  0.407      between-refit; mixed, not added
    residual corr length (dec)    0.073      variogram, squared-exponential
    gamma0, gamma_slope (ln)      0.045      between-refit, median|d ln| 0.067, capped at 1.5x
    floor_mean_db                 1.066      between-clip (pooled over 4 fits)
    floor_tilt_db_oct             0.210      between-refit
    floor_shape_db (per point)    2.175      between-refit, AR(1) lag-1 0.838 across points
    mic_gain_db                   1.122      between-refit
    mic_floor_db                  0.199      between-refit
    gain_all_db                   0.127      between-refit
    coherence_k_half (ln)         0.198      between-clip (pooled over 4 fits)
    amp_exp                       2.733      between-refit
    floor_exp                     2.733      between-refit of amp_exp, see below
    floor_static_rel (ln)         0.332      between-refit

Two quantities need a stated exception:

* ``floor_exp``: its own between-refit spread is 3.53 and includes a
  ``+6.79 -> -2.71`` SIGN FLIP between two fits of the same FLY125 audio. That
  is an identifiability artefact of a fit taken at one operating point, not a
  neighbourhood, so ``amp_exp``'s narrower between-refit width is used for both
  speed exponents. A sampled ``floor_exp`` is then HARD-CLAMPED to ``>= 0``,
  because the DREGON room-2 refined fit sits at ``-3.711`` and a negative floor
  exponent makes the broadband floor DIVERGE as the rotors stop: one telemetry
  sample of exactly zero rev/s renders ``inf``. The clamp preserves the floor
  level at the anchor's own speed, the same two lines
  ``flight_model._rebase_divergent_floor`` uses, reimplemented locally in
  :func:`_rebase_divergent_floor` (that module is owned elsewhere).
* ``amp_exp``/``floor_exp`` widths look large in exponent units but are small
  in level: every anchor here runs at 75-90 rev/s against the model's 80 rev/s
  reference, so one unit of exponent is only ``10*log10(rps/80) = 0.3`` dB of
  level at the anchor speed. The perturbation changes the SPEED LAW (how the
  rig responds away from its fitted operating point) and barely moves the
  fitted-point spectrum, which is exactly the intended freedom.

Guards
------
:func:`check_sample` returns every guard result and :func:`sample_rig` applies
it internally, redrawing up to ``max_attempts`` times and raising
:class:`SampleRejected` rather than returning an invalid export:

* ``finite`` — the model PSD (the render's expectation, 9 ms instead of 3 s) is
  finite and positive at the anchor speed AND at exactly zero speed, which is
  the divergent-floor trap.
* ``trend_falls`` — the sampled trend must be lower at ``k = K`` than at
  ``k = 1`` by at least ``TREND_MARGIN_DB = 3`` dB, per rotor. Measured: the
  smallest total trend drop over the 120 CRUISE rotor profiles is 15.0 dB, so
  the guard admits every measured cruise rotor with 12 dB to spare. The single
  measured profile that fails it is a FLY125 STANDBY rotor (a +7.6 dB RISE
  across the band) — a different regime, deliberately outside this guard.
* ``parity_sign`` — the low-band parity amplitude must stay positive. The
  parity scale is drawn lognormal, so inversion is structurally impossible;
  the guard is checked anyway and fires only if an anchor is handed in whose
  own parity is already inverted. Measured: 0 of 124 rotor profiles has a
  negative low-band parity amplitude (minimum 1.27 dB).
* ``gamma_excursion`` — total linewidth excursion capped at 1.5x. Stated, not
  measured: it keeps a sampled linewidth in the anchor's own decade, far from
  the 3.6x jump the DREGON cruise-to-flight pair shows and the 14.7x
  between-rig span.
* ``ltas`` — RMS 1/3-octave LTAS deviation from the reference export (the
  anchor for :func:`sample_rig`, the INTERPOLATED point for
  :func:`sample_path`), inside :data:`LEVEL_BAND_HZ`. Tolerance
  ``LTAS_ENVELOPE_X * REAL_PAIR_RMS_DB``, or a caller's ``ltas_tol``,
  STRENGTH-INDEPENDENT. This guard rejects IMPLAUSIBLE rigs, not distant ones:
  its envelope is twice the measured difference between two REAL drones of
  this kind. See :data:`LTAS_ENVELOPE_X` for the full reasoning chain and for
  why the worst single band is reported but not guarded.

In :func:`sample_path` the two structural invariants also hold at the
interpolated point BY CONSTRUCTION, with no guard needed: a convex combination
of two positive parity envelopes is positive, and of two negative trend slopes
is negative. Only the perturbation on top can break them, and it is rejected
when it does.

CLI
---
``PYTHONPATH=src python -m experiments.stochastic_fit.rig_sampler`` runs phase A
over all six fits, writes ``results/rig_sampler/structure.json`` and prints the
table plus a frozen-against-measured comparison.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# provenance: the fits phase A measures
# ---------------------------------------------------------------------------

#: renderer reference speed, ``srn.StochasticParams.amp_rps_ref``
AMP_RPS_REF = 80.0

FIT_PATHS: dict[str, str] = {
    "michael_cruise": "results/S2/cruise_8clip.json",
    "michael_cruise_refined": "results/S2/cruise_8clip_refined.json",
    "michael_cruise_dyn": "results/S2/cruise_8clip_dyn.json",
    "michael_standby": "results/S2/standby.json",
    "dregon_cruise_refined": "results/S2/dregon_room2_cruise_refined.json",
    "dregon_flight": "results/S2/dregon_flight.json",
}

#: fits of the SAME rig and the same recordings, different fit configuration
REFIT_GROUPS: dict[str, tuple[str, ...]] = {
    "michael_cruise_rig": ("michael_cruise", "michael_cruise_refined", "michael_cruise_dyn"),
    "dregon_cruise_rig": ("dregon_cruise_refined", "dregon_flight"),
}

#: fits with more than one clip, i.e. the ones that carry a between-clip spread
MULTICLIP_FITS = (
    "michael_cruise",
    "michael_cruise_refined",
    "michael_cruise_dyn",
    "dregon_cruise_refined",
)

#: the cruise family the trend guard is calibrated on (standby is another regime)
CRUISE_FITS = (
    "michael_cruise",
    "michael_cruise_refined",
    "michael_cruise_dyn",
    "dregon_cruise_refined",
    "dregon_flight",
)

STRUCTURE_PATH = Path("results/rig_sampler/structure.json")

#: half-width, in decades of order, of the parity-envelope smoother
PARITY_SMOOTH_DEC = 0.12
#: orders averaged for the reported "low band" parity amplitude
PARITY_LOW_ORDERS = 16

# ---------------------------------------------------------------------------
# phase A: the decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileParts:
    """One rotor's profile split into gain, trend, parity envelope, residual.

    ``profile == gain + slope * (x - x.mean()) + env * s + resid`` exactly, so
    the sampler can perturb the parts and reassemble without drift.
    """

    x: np.ndarray  # (K,) log10 order
    s: np.ndarray  # (K,) +1 even, -1 odd
    gain: float  # mean level over the order grid, dB
    slope: float  # dB per decade of order
    env: np.ndarray  # (K,) parity amplitude envelope, dB
    resid: np.ndarray  # (K,) what is left
    profile: np.ndarray  # (K,) the input, kept for exact reassembly

    @property
    def trend_drop_db(self) -> float:
        """How far the trend falls from ``k = 1`` to ``k = K``. Positive = falls."""
        return float(-self.slope * (self.x[-1] - self.x[0]))

    @property
    def parity_low_db(self) -> float:
        return float(self.env[:PARITY_LOW_ORDERS].mean())

    @property
    def resid_std(self) -> float:
        # the last order is a band-edge artefact of the fit (the FLY125 cruise
        # profile jumps +12.5 dB on its final order, where the fit has the
        # fewest cells); it is excluded from every measured statistic and left
        # untouched by the sampler's perturbation
        return float(self.resid[:-1].std(ddof=2))


def parity_sign(n_orders: int) -> np.ndarray:
    """``+1`` on even orders, ``-1`` on odd, for ``k = 1..n_orders``."""
    k = np.arange(1, n_orders + 1)
    return np.where(k % 2 == 0, 1.0, -1.0)


def _log_smooth(y: np.ndarray, x: np.ndarray, half_width: float) -> np.ndarray:
    """Moving average of ``y`` over ``|x_j - x_i| <= half_width``."""
    out = np.empty_like(y)
    for i in range(y.size):
        out[i] = y[np.abs(x - x[i]) <= half_width].mean()
    return out


def decompose_profile(profile_db: np.ndarray) -> ProfileParts:
    """Split one rotor's ``(K,)`` profile into gain, log-order trend, parity, residual.

    The parity envelope comes from the 3-point alternating filter
    ``p[k] - (p[k-1] + p[k+1]) / 2``, which equals ``2 * A * s(k)`` for a pure
    alternating component of amplitude ``A``, smoothed over
    :data:`PARITY_SMOOTH_DEC` decades so the envelope can decay with order the
    way the measured profiles do. The trend is then least-squares fitted to the
    parity-free part, with the order axis centred so that the slope and the
    gain are orthogonal: perturbing the slope does not move the level.
    """
    p = np.asarray(profile_db, dtype=np.float64).ravel()
    n = p.size
    x = np.log10(np.arange(1, n + 1, dtype=np.float64))
    s = parity_sign(n)
    alt = np.empty(n)
    alt[1:-1] = s[1:-1] * (p[1:-1] - 0.5 * (p[:-2] + p[2:])) / 2.0
    alt[0] = alt[1]
    alt[-1] = alt[-2]
    env = _log_smooth(alt, x, PARITY_SMOOTH_DEC)
    smooth = p - env * s
    design = np.stack([np.ones(n), x - x.mean()], axis=1)
    coef, *_ = np.linalg.lstsq(design, smooth, rcond=None)
    resid = smooth - design @ coef
    return ProfileParts(
        x=x,
        s=s,
        gain=float(coef[0]),
        slope=float(coef[1]),
        env=env,
        resid=resid,
        profile=p,
    )


def ols_forms(profile_db: np.ndarray) -> dict[str, dict[str, Any]]:
    """The two requested closed-form designs, for the comparison phase A owes.

    ``A`` is ``[1, x, s]`` — one trend plus one parity amplitude. ``B`` is
    ``[1, x, s, s*x]`` — separate even and odd trends.
    """
    p = np.asarray(profile_db, dtype=np.float64).ravel()
    n = p.size
    x = np.log10(np.arange(1, n + 1, dtype=np.float64))
    xc = x - x.mean()
    s = parity_sign(n)
    out: dict[str, dict[str, Any]] = {}
    for tag, cols in (
        ("one_trend_plus_parity", [np.ones(n), xc, s]),
        ("even_odd_trends", [np.ones(n), xc, s, s * xc]),
    ):
        design = np.stack(cols, axis=1)
        coef, *_ = np.linalg.lstsq(design, p, rcond=None)
        out[tag] = {
            "resid_std": float((p - design @ coef)[:-1].std(ddof=len(cols))),
            "coef": [float(v) for v in coef],
        }
    return out


def effective_profile(export: dict[str, Any]) -> np.ndarray:
    """``(R, K)`` static profile the renderer actually uses.

    ``params_from_export`` adds the time-mean of the dynamic ``h_db`` term to
    ``profile_db``, so the decomposition must see the sum. Every fit measured
    here has ``h_db == 0``, but the sum is what is correct in general.
    """
    p = np.atleast_2d(np.asarray(export["profile_db"], dtype=np.float64))
    h = np.asarray(export.get("h_db", 0.0), dtype=np.float64)
    if h.ndim == 3:
        p = p + h.mean(axis=-1)[: p.shape[0]]
    return p


# ---------------------------------------------------------------------------
# phase A: residual correlation structure
# ---------------------------------------------------------------------------


def _acf_bins(resid: np.ndarray, x: np.ndarray, n_bins: int, max_lag: float):
    r = resid - resid.mean()
    var = float((r * r).mean())
    iu = np.triu_indices(r.size, 1)
    lag = np.abs(x[:, None] - x[None, :])[iu]
    prod = (r[:, None] * r[None, :])[iu]
    edges = np.linspace(0.0, max_lag, n_bins + 1)
    cen, cor = [], []
    for i in range(n_bins):
        m = (lag >= edges[i]) & (lag < edges[i + 1])
        if m.sum() < 8:
            continue
        cen.append(lag[m].mean())
        cor.append(prod[m].mean() / max(var, 1e-12))
    return np.asarray(cen), np.asarray(cor), lag, prod


_LENGTH_GRID = np.geomspace(0.005, 1.0, 400)


def _best_length(cen: np.ndarray, cor: np.ndarray, kind: str) -> tuple[float, float]:
    lags = cen[None, :]
    if kind == "exp":
        model = np.exp(-lags / _LENGTH_GRID[:, None])
    else:
        model = np.exp(-0.5 * (lags / _LENGTH_GRID[:, None]) ** 2)
    err = np.mean((model - cor[None, :]) ** 2, axis=1)
    i = int(np.argmin(err))
    return float(_LENGTH_GRID[i]), float(err[i])


def residual_correlation(resid: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """Correlation length of the profile residual in ``log10`` order.

    Both kernels are fitted only up to the autocorrelation's first zero
    crossing. Beyond it the empirical ACF of a single detrended realisation is
    driven negative by the mean and trend removal — it reaches -0.6 on FLY125
    cruise rotor 0 — and no positive kernel can fit that. The short-lag
    variogram ``Var(dr) = 2 sigma^2 (1 - rho(h))`` is reported too because it
    does not carry that bias, and it is what the sampler's width uses.
    """
    cen, cor, lag, _ = _acf_bins(resid, x, n_bins=24, max_lag=0.8)
    zero = float(cen[-1])
    for i in range(cor.size - 1):
        if cor[i] > 0.0 >= cor[i + 1]:
            zero = float(np.interp(0.0, [cor[i + 1], cor[i]], [cen[i + 1], cen[i]]))
            break
    keep = cen <= zero
    out: dict[str, float] = {"zero_crossing_dec": zero}
    for kind in ("exp", "sqexp"):
        length, err = _best_length(cen[keep], cor[keep], kind)
        out[f"length_{kind}_dec"] = length
        out[f"err_{kind}"] = err
    iu = np.triu_indices(resid.size, 1)
    diff2 = ((resid[:, None] - resid[None, :]) ** 2)[iu]
    near = lag <= zero
    var = float(np.var(resid))
    pred = 2.0 * var * (1.0 - np.exp(-0.5 * (lag[near][None, :] / _LENGTH_GRID[:, None]) ** 2))
    err = np.mean((pred - diff2[near][None, :]) ** 2, axis=1)
    out["length_variogram_dec"] = float(_LENGTH_GRID[int(np.argmin(err))])
    return out


#: work grid the demo's renders run on (``stage2.render_from_export``'s default,
#: ``clips.NATIVE_SR``). The declared rate term is tied to it: +4.4032 dB.
RENDER_SAMPLE_RATE_WORK = 44100


def to_physical(
    entry: Mapping[str, Any], *, sample_rate_work: int = RENDER_SAMPLE_RATE_WORK
) -> dict[str, Any]:
    """One clip's exported ``params`` on the PHYSICAL level scale.

    BOTH halves of the declared conversion, in order:

    1. :func:`revised_eval._to_physical` folds the clip's own
       ``scores.power_scale`` into ``revised_eval.ABSOLUTE_POWER_DB_FIELDS``
       (``profile_db``, ``floor_mean_db``). A stage-2 fit works on its clip's
       periodogram divided by that scale and exports levels in THAT unit, so
       the raw ``params`` dict is not comparable in absolute level with any
       other clip's. Measured over the six fits: -22.31 dB (DREGON room-2
       cruise clip 0) to -39.59 dB (FLY125 standby), varying BETWEEN CLIPS of
       one fit by 0.45-1.04 dB, which is why it moves phase A's between-clip
       level widths and not only the absolute level.
    2. :func:`revised_eval.to_renderer_units` adds
       ``10 log10(work / analysis)`` = +4.4032 dB at the 44.1 kHz work grid the
       renders here use. Valid only with ``normalize_rms=None``, which is what
       every level measurement in this module passes.

    Step 2 was WRONGLY OMITTED in an earlier revision of this module, and the
    correction is worth recording because the mistake was subtle. The omission
    was justified on a whole-array RMS agreement: with power_scale folded and
    no rate term the FLY125 cruise anchor's render RMS sat 0.43 dB from its
    real clip, and adding 4.40 dB appeared to break it. That scalar was
    CONFOUNDED — two errors cancelling. Band-resolved on the same anchor
    (FLY125 [32, 48) s, 8 mics): without the term 50-100 Hz reads +9.61 dB,
    100-200 +2.80, 300-700 -1.88, 1500-3000 -2.72, 5000-7900 -3.84; with it,
    +14.02, +7.20, +2.52, +1.69, +0.56. Above 300 Hz the mean absolute
    deviation is 2.69 dB WITHOUT the term and 1.72 dB WITH it, and without it
    the model is systematically low and tilted. The apparent RMS agreement was
    the 50-200 Hz excess paying for the missing 4.4 dB. The term is the
    declared conversion, validated against a planted tone and a planted floor,
    so it is applied; the residual 50-200 Hz excess is a separate, localised
    defect (concentrated in orders k=1-3, the floor between lines being right
    to ~1 dB) under investigation elsewhere, and it is why this module's level
    coordinate is BAND-RESTRICTED — see :data:`LEVEL_BAND_HZ`.
    """
    from experiments.stochastic_fit import revised_eval

    folded = revised_eval._to_physical(entry)
    mp = revised_eval.ModelParams(params=folded, spec={}, source={})
    out = dict(revised_eval.to_renderer_units(mp, sample_rate_work=int(sample_rate_work)).params)
    out["rate_factor_db"] = float(revised_eval.rate_factor_db(int(sample_rate_work)))
    return out


def load_anchor(fit_path: Path | str, selector: Any, *, physical: bool = True) -> dict[str, Any]:
    """One clip's export from a stage-2 fit summary, on the physical scale.

    ``selector`` is an integer index or a clip-id prefix, the same convention as
    ``scripts/_revised_ab_render.prev_export`` — which returns the RAW
    ``params`` dict. Every anchor this module samples from goes through here so
    that no scaled-unit level reaches a decomposition or a level coordinate;
    ``physical=False`` returns the raw dict and exists only to measure what
    folding changed.
    """
    summary = json.loads(Path(fit_path).read_text())
    clips = summary["clips"]
    if isinstance(selector, int):
        entry = list(clips.values())[selector]
    else:
        entry = next((e for cid, e in clips.items() if cid.startswith(str(selector))), None)
        if entry is None:
            raise KeyError(f"{fit_path}: no clip starts with {selector!r}")
    return to_physical(entry) if physical else dict(entry["params"])


# ---------------------------------------------------------------------------
# phase A: measurement over all fits
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _scalar(params: dict[str, Any], key: str) -> float:
    v = np.asarray(params[key], dtype=np.float64)
    return float(v.mean()) if v.ndim else float(v)


def _vector(params: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(params[key], dtype=np.float64).ravel()


def third_octave(freqs: np.ndarray, level_db: np.ndarray, f_lo=50.0, f_hi=7900.0):
    """``(centres, band levels)`` — power-averaged 1/3-octave bands."""
    edges = 2.0 ** np.arange(np.log2(f_lo), np.log2(f_hi) + 1e-9, 1.0 / 3.0)
    power = 10.0 ** (np.asarray(level_db, dtype=np.float64) / 10.0)
    freqs = np.asarray(freqs, dtype=np.float64)
    cen, out = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs < hi)
        if not m.any():
            continue
        cen.append(float(np.sqrt(lo * hi)))
        out.append(float(10.0 * np.log10(max(power[m].mean(), 1e-300))))
    return np.asarray(cen), np.asarray(out)


LTAS_FREQS = np.linspace(0.0, 8000.0, 2049)

#: analysis length for :func:`audio_bands`, 0.512 s at 16 kHz
BAND_NFFT = 8192


def audio_bands(x: np.ndarray, n: int = BAND_NFFT) -> tuple[np.ndarray, np.ndarray]:
    """``(centres, 1/3-octave levels)`` of a real or rendered signal, mic mean.

    Welch-averaged Hann periodogram, power-averaged over channels and then over
    each band. Absolute dB: on the physical level path (``power_scale`` folded,
    ``normalize_rms=None``) a render's bands and a real clip's bands are
    directly comparable, which is the only way the coverage question can be
    asked.
    """
    xx = np.atleast_2d(np.asarray(x, dtype=np.float64))
    w = np.hanning(n + 1)[:n]
    power = np.zeros(n // 2 + 1)
    for ch in xx:
        frames = np.stack([ch[s : s + n] * w for s in range(0, ch.size - n, n // 2)])
        power += (np.abs(np.fft.rfft(frames, axis=-1)) ** 2).mean(axis=0)
    power /= xx.shape[0]
    freqs = np.fft.rfftfreq(n, 1.0 / 16000)
    return third_octave(freqs, 10.0 * np.log10(power + 1e-300))


def ltas_vs_real(
    export: dict[str, Any],
    rps: np.ndarray,
    audio: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, float]:
    """How far an export's own render sits from the REAL clip, in 1/3 octaves.

    This is what the re-scoped LTAS guard's envelope is calibrated on: a
    transfer arm has to be able to reach real data, so the plausibility
    envelope is a stated multiple of THIS, not of the between-clip spread.
    Absolute (not level-matched), which requires the physical level path.

    ``rms_db``/``band_max_db``/``level_db`` are restricted to
    :data:`LEVEL_BAND_HZ`, which is what the guard uses; the ``*_full_band``
    entries cover 50 Hz - 7.9 kHz and are dominated by the known k=1-3 render
    excess, so they are reported and not guarded on.
    """
    from experiments.stochastic_fit import stage2

    render = stage2.render_from_export(export, rps, seed=seed, normalize_rms=None)
    cen, band = audio_bands(render)
    _, real = audio_bands(audio)
    dev = real - band
    keep = (cen >= LEVEL_BAND_HZ[0]) & (cen <= LEVEL_BAND_HZ[1])
    return {
        "rms_db": float(np.sqrt((dev[keep] ** 2).mean())),
        "band_max_db": float(np.abs(dev[keep]).max()),
        "level_db": float(dev[keep].mean()),
        "rms_db_full_band": float(np.sqrt((dev**2).mean())),
        "band_max_db_full_band": float(np.abs(dev).max()),
        "level_db_full_band": float(dev.mean()),
    }


def nominal_rates(export: dict[str, Any]) -> np.ndarray:
    """``(R,)`` per-rotor rev/s the export itself declares.

    ``carrier`` is the fitted per-rotor fundamental rate on the clip's knot
    grid, so the export carries its own operating point and the guards need no
    telemetry to evaluate a sample.
    """
    return np.asarray(export["carrier"], dtype=np.float64).mean(axis=-1)


def model_ltas(export: dict[str, Any], rates: np.ndarray, freqs: np.ndarray = LTAS_FREQS):
    """Model long-term spectrum in dB: the render's EXPECTATION, not a render.

    The renderer's own :func:`data_processing.stochastic_rotor_noise.build_psd`
    on a single frame at ``rates``, floor plus every rotor's line power. The
    per-microphone gains are deliberately left out: they enter as one constant
    per channel and are perturbed separately. 9 ms against 3 s for a 4 s
    render, which is what lets the guards run inside the redraw loop.
    """
    from data_processing import stochastic_rotor_noise as srn
    from experiments.stochastic_fit import stage2

    rates = np.asarray(rates, dtype=np.float64)
    params = stage2.params_from_export(export, rates, sample_rate=32000, n_mics=8)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        psd = srn.build_psd(
            params,
            rates[:, None],
            np.asarray(freqs, dtype=np.float64),
            dt=0.1,
            rng=np.random.default_rng(0),
        )
        total = psd["floor"][0] + psd["lines"][:, 0, :].sum(axis=0)
        return 10.0 * np.log10(np.maximum(total, 1e-300)), total


def measure_structure(root: Path | str = ".") -> dict[str, Any]:
    """Phase A. Measure every spread the sampler needs, over every listed fit.

    Returns the dict written to ``results/rig_sampler/structure.json``.
    """
    root = Path(root)
    fits: dict[str, dict[str, Any]] = {}
    provenance: dict[str, Any] = {}
    for name, rel in FIT_PATHS.items():
        path = root / rel
        fits[name] = json.loads(path.read_text())
        # EVERY clip is put on the physical level scale here, once, so no
        # measurement below can see a scaled-unit level
        scales = {}
        for cid, entry in fits[name]["clips"].items():
            entry["params"] = to_physical(entry)
            scales[cid] = entry["params"]["power_scale_folded_db"]
        first = next(iter(fits[name]["clips"].values()))["params"]
        provenance[name] = {
            "path": rel,
            "sha256": _sha256(path),
            "n_clips": len(fits[name]["clips"]),
            "n_rotors": int(np.shape(first["profile_db"])[0]),
            "n_orders": int(np.shape(first["profile_db"])[1]),
            "power_scale_folded_db": scales,
        }

    # --- per rotor-profile decomposition over every clip of every fit -------
    per_profile: list[dict[str, Any]] = []
    decomposed: dict[str, dict[str, list[ProfileParts]]] = {}
    for name, fit in fits.items():
        decomposed[name] = {}
        for cid, clip in fit["clips"].items():
            prof = effective_profile(clip["params"])
            parts = [decompose_profile(prof[r]) for r in range(prof.shape[0])]
            decomposed[name][cid] = parts
            forms = [ols_forms(prof[r]) for r in range(prof.shape[0])]
            for r, (p, f) in enumerate(zip(parts, forms)):
                corr = residual_correlation(p.resid[:-1], p.x[:-1])
                per_profile.append(
                    {
                        "fit": name,
                        "clip": cid,
                        "rotor": r,
                        "gain_db": p.gain,
                        "slope_db_dec": p.slope,
                        "slope_db_dec_ols": f["one_trend_plus_parity"]["coef"][1],
                        "parity_low_db": p.parity_low_db,
                        "parity_ols_db": f["one_trend_plus_parity"]["coef"][2],
                        "parity_neg_frac": float((p.env < 0).mean()),
                        "resid_std_db": p.resid_std,
                        "resid_std_one_trend": f["one_trend_plus_parity"]["resid_std"],
                        "resid_std_even_odd": f["even_odd_trends"]["resid_std"],
                        "trend_drop_db": p.trend_drop_db,
                        **corr,
                    }
                )

    def col(key: str, rows=per_profile) -> np.ndarray:
        return np.asarray([r[key] for r in rows], dtype=np.float64)

    decomp_compare = {
        "n_rotor_profiles": len(per_profile),
        "resid_std_envelope_mean": float(col("resid_std_db").mean()),
        "resid_std_one_trend_plus_parity_mean": float(col("resid_std_one_trend").mean()),
        "resid_std_even_odd_trends_mean": float(col("resid_std_even_odd").mean()),
        "envelope_wins_over_even_odd": int((col("resid_std_db") < col("resid_std_even_odd")).sum()),
        "even_odd_wins_over_one_trend": int(
            (col("resid_std_even_odd") < col("resid_std_one_trend")).sum()
        ),
        "note": (
            "the single-trend design [1, x, s] leaves 6.33 dB and is the form phase A "
            "was asked for; separate even/odd trends [1, x, s, s*x] improve it to "
            "5.50 dB, and the measured decaying parity envelope improves it further to "
            "5.08 dB, so the sampler uses the envelope form"
        ),
    }

    resid_corr = {
        "kernel_used": "sqexp",
        "length_sqexp_dec_median": float(np.median(col("length_sqexp_dec"))),
        "length_exp_dec_median": float(np.median(col("length_exp_dec"))),
        "length_variogram_dec_median": float(np.median(col("length_variogram_dec"))),
        "err_sqexp_mean": float(col("err_sqexp").mean()),
        "err_exp_mean": float(col("err_exp").mean()),
        "sqexp_better_than_exp": int((col("err_sqexp") < col("err_exp")).sum()),
        "zero_crossing_dec_mean": float(col("zero_crossing_dec").mean()),
        "resid_std_db_mean": float(col("resid_std_db").mean()),
    }

    parity = {
        "low_band_db_mean": float(col("parity_low_db").mean()),
        "low_band_db_min": float(col("parity_low_db").min()),
        "low_band_db_max": float(col("parity_low_db").max()),
        "n_inverted_low_band": int((col("parity_low_db") <= 0.0).sum()),
        "envelope_negative_frac_mean": float(col("parity_neg_frac").mean()),
        "note": (
            "0 of the measured rotor profiles has an inverted low-band parity split; "
            "the negative envelope fraction is confined to high orders where the "
            "split has already decayed to ~0 dB"
        ),
    }

    cruise_rows = [r for r in per_profile if r["fit"] in CRUISE_FITS]
    trend = {
        "drop_db_min_all": float(col("trend_drop_db").min()),
        "drop_db_min_cruise": float(col("trend_drop_db", cruise_rows).min()),
        "drop_db_q05": float(np.quantile(col("trend_drop_db"), 0.05)),
        "drop_db_median": float(np.median(col("trend_drop_db"))),
        "n_rising": int((col("trend_drop_db") < 0.0).sum()),
        "rising_profiles": [
            f"{r['fit']}:{r['clip']}:r{r['rotor']}" for r in per_profile if r["trend_drop_db"] < 0.0
        ],
    }

    # --- tier 1: between clips of one fit ----------------------------------
    between_clip: dict[str, Any] = {}
    gain_stds, fmean_stds, coh_ln_stds = [], [], []
    for name in MULTICLIP_FITS:
        clips = list(fits[name]["clips"].values())
        gains = np.stack([effective_profile(c["params"]).mean(axis=-1) for c in clips])
        gain_stds.append(gains.std(axis=0, ddof=1))
        fmean_stds.append(
            float(np.std([_scalar(c["params"], "floor_mean_db") for c in clips], ddof=1))
        )
        coh_ln_stds.append(
            float(np.std(np.log([_scalar(c["params"], "coherence_k_half") for c in clips]), ddof=1))
        )
    gain_std = np.concatenate(gain_stds)
    between_clip["rotor_gain_db"] = {
        "per_fit_per_rotor_std": [[float(v) for v in s] for s in gain_stds],
        "pooled_rms": float(np.sqrt((gain_std**2).mean())),
        "median": float(np.median(gain_std)),
        "max": float(gain_std.max()),
        "n": int(gain_std.size),
    }
    between_clip["floor_mean_db"] = {
        "per_fit_std": fmean_stds,
        "pooled_rms": float(np.sqrt(np.mean(np.square(fmean_stds)))),
    }
    between_clip["coherence_k_half_ln"] = {
        "per_fit_std": coh_ln_stds,
        "pooled_rms": float(np.sqrt(np.mean(np.square(coh_ln_stds)))),
    }
    # Main's follow-up: does the level width move again on the BAND-RESTRICTED
    # coordinate? Measured on the model band level (no telemetry needed: each
    # clip's own carrier), restricted to LEVEL_BAND_HZ, so it is the same
    # coordinate the path sampler carries.
    band_lvl_stds = []
    for name in MULTICLIP_FITS:
        lv = []
        for c in fits[name]["clips"].values():
            level, _ = model_ltas(c["params"], nominal_rates(c["params"]))
            cen_b, bands_b = third_octave(LTAS_FREQS, level)
            m = (cen_b >= LEVEL_BAND_HZ[0]) & (cen_b <= LEVEL_BAND_HZ[1])
            lv.append(float(10.0 * np.log10(np.mean(10.0 ** (bands_b[m] / 10.0)))))
        band_lvl_stds.append(float(np.std(lv, ddof=1)))
    between_clip["band_level_db"] = {
        "per_fit_std": band_lvl_stds,
        "pooled_rms": float(np.sqrt(np.mean(np.square(band_lvl_stds)))),
        "band_hz": list(LEVEL_BAND_HZ),
        "note": (
            "between-clip spread of the model level in LEVEL_BAND_HZ, each clip at its own "
            "carrier. This is the level coordinate the path sampler carries; compare with "
            "floor_mean_db.pooled_rms, which is the FIELD's spread"
        ),
    }
    # which fields move between clips at all
    tied, free = [], []
    ref = next(iter(fits["michael_cruise"]["clips"].values()))["params"]
    for key in ref:
        stack = []
        for c in fits["michael_cruise"]["clips"].values():
            stack.append(np.asarray(c["params"][key], dtype=np.float64))
        arr = np.stack(stack)
        (free if arr.std(axis=0).max() > 1e-9 else tied).append(key)
    between_clip["fields_varying_between_clips"] = free
    between_clip["fields_tied_between_clips"] = tied
    between_clip["note"] = (
        "profile_db varies between clips ONLY by a per-rotor constant (the residual "
        "shape spread across clips is 1e-6 dB), so the between-clip profile freedom IS "
        "the per-rotor gain; every tied field has identically zero between-clip spread "
        "and needs the between-refit fallback"
    )

    # --- tier 2: between refits of one rig ---------------------------------
    scalar_keys = (
        "gamma0",
        "gamma_slope",
        "coherence_k_half",
        "floor_mean_db",
        "floor_tilt_db_oct",
        "amp_exp",
        "floor_exp",
        "floor_static_rel",
    )
    vector_keys = ("floor_shape_db", "mic_gain_db", "mic_floor_db", "gain_all_db")
    pairs = [(a, b) for names in REFIT_GROUPS.values() for a, b in itertools.combinations(names, 2)]
    between_refit: dict[str, Any] = {"pairs": [f"{a}|{b}" for a, b in pairs]}
    for key in scalar_keys:
        deltas = [
            abs(
                _scalar(next(iter(fits[a]["clips"].values()))["params"], key)
                - _scalar(next(iter(fits[b]["clips"].values()))["params"], key)
            )
            for a, b in pairs
        ]
        between_refit[key] = {
            "abs_delta": deltas,
            "median_abs_delta": float(np.median(deltas)),
            "sigma": float(np.median(deltas) / np.sqrt(2.0)),
        }
    for key in vector_keys:
        deltas, ar1 = [], []
        for a, b in pairs:
            va = _vector(next(iter(fits[a]["clips"].values()))["params"], key)
            vb = _vector(next(iter(fits[b]["clips"].values()))["params"], key)
            n = min(va.size, vb.size)
            d = va[:n] - vb[:n]
            if key == "floor_shape_db":
                # the control points' common level is degenerate with
                # floor_mean_db, so the shape width is measured mean-removed
                d = d - d.mean()
                ar1.append(float(np.corrcoef(d[:-1], d[1:])[0, 1]))
            deltas.append(float(np.sqrt(np.mean(d**2))))
        between_refit[key] = {
            "rms_delta": deltas,
            "median_rms_delta": float(np.median(deltas)),
            "sigma": float(np.median(deltas) / np.sqrt(2.0)),
        }
        if ar1:
            between_refit[key]["lag1_corr_mean"] = float(np.mean(ar1))
    # linewidth, as a log ratio, over both linewidth parameters
    gamma_ln = []
    for a, b in pairs:
        for key in ("gamma0", "gamma_slope"):
            va = _scalar(next(iter(fits[a]["clips"].values()))["params"], key)
            vb = _scalar(next(iter(fits[b]["clips"].values()))["params"], key)
            if min(va, vb) > 0.01:  # FLY125 gamma0 is numerically zero
                gamma_ln.append(
                    {"pair": f"{a}|{b}", "key": key, "abs_dln": float(abs(np.log(va / vb)))}
                )
    between_refit["gamma_ln"] = {
        "samples": gamma_ln,
        "median_abs_dln": float(np.median([g["abs_dln"] for g in gamma_ln])),
        "sigma": float(np.median([g["abs_dln"] for g in gamma_ln]) / np.sqrt(2.0)),
        "note": (
            "the dregon cruise|flight gamma_slope ratio is 3.6x (|dln| 1.28), a regime "
            "change rather than a neighbourhood; the median rejects it"
        ),
    }
    static_ln = []
    for a, b in pairs:
        va = max(_scalar(next(iter(fits[a]["clips"].values()))["params"], "floor_static_rel"), 1e-9)
        vb = max(_scalar(next(iter(fits[b]["clips"].values()))["params"], "floor_static_rel"), 1e-9)
        static_ln.append(float(abs(np.log(va / vb))))
    between_refit["floor_static_rel_ln"] = {
        "abs_dln": static_ln,
        "median_abs_dln": float(np.median(static_ln)),
        "sigma": float(np.median(static_ln) / np.sqrt(2.0)),
    }
    # profile shape, clip- and rotor-matched across refits
    slope_d, parity_d, resid_frac = [], [], []
    for a, b in pairs:
        for cid in sorted(set(decomposed[a]) & set(decomposed[b])):
            for pa, pb in zip(decomposed[a][cid], decomposed[b][cid]):
                n = min(pa.resid.size, pb.resid.size) - 1
                slope_d.append(abs(pa.slope - pb.slope))
                parity_d.append(abs(np.log(pa.parity_low_db / pb.parity_low_db)))
                resid_frac.append(
                    float((pa.resid[:n] - pb.resid[:n]).std(ddof=1))
                    / float(pa.resid[:n].std(ddof=1))
                )
    between_refit["slope_db_dec"] = {
        "n": len(slope_d),
        "median_abs_delta": float(np.median(slope_d)),
        "sigma": float(np.median(slope_d) / np.sqrt(2.0)),
    }
    between_refit["parity_ln"] = {
        "n": len(parity_d),
        "median_abs_dln": float(np.median(parity_d)),
        "sigma": float(np.median(parity_d) / np.sqrt(2.0)),
    }
    between_refit["resid_redraw_frac"] = {
        "n": len(resid_frac),
        "median": float(np.median(resid_frac)),
        "note": (
            "std(resid_a - resid_b) / std(resid_a); used DIRECTLY as the mixing "
            "amplitude, since resid' = rho resid + sqrt(1-rho^2) sigma z gives "
            "std(resid' - resid) / sigma = sqrt(2 (1 - rho))"
        ),
    }

    # --- tier 3: between rigs / regimes ------------------------------------
    rig_means: dict[str, dict[str, float]] = {}
    for name, fit in fits.items():
        first = next(iter(fit["clips"].values()))["params"]
        rows = [r for r in per_profile if r["fit"] == name]
        rig_means[name] = {
            "slope_db_dec": float(col("slope_db_dec", rows).mean()),
            "slope_db_dec_ols": float(col("slope_db_dec_ols", rows).mean()),
            "parity_low_db": float(col("parity_low_db", rows).mean()),
            "resid_std_db": float(col("resid_std_db", rows).mean()),
            "gain_db": float(col("gain_db", rows).mean()),
            **{k: _scalar(first, k) for k in scalar_keys},
        }
    between_rig = {
        "per_fit": rig_means,
        "std_of_fit_means": {
            k: float(np.std([rig_means[n][k] for n in rig_means], ddof=1))
            for k in next(iter(rig_means.values()))
        },
        "between_rotor_within_fit": {},
    }
    for key in ("gain_db", "slope_db_dec", "slope_db_dec_ols", "parity_low_db"):
        stds = []
        for name in fits:
            for cid in decomposed[name]:
                rows = [r for r in per_profile if r["fit"] == name and r["clip"] == cid]
                stds.append(float(col(key, rows).std(ddof=1)))
        between_rig["between_rotor_within_fit"][key] = float(np.mean(stds))
    ln_env = []
    for name in fits:
        for cid in decomposed[name]:
            rows = [r for r in per_profile if r["fit"] == name and r["clip"] == cid]
            ln_env.append(float(np.log(col("parity_low_db", rows)).std(ddof=1)))
    between_rig["between_rotor_within_fit"]["parity_ln"] = float(np.mean(ln_env))

    # --- between-clip LTAS spread: the plausibility envelope ---------------
    ltas_dev = []
    for name in MULTICLIP_FITS:
        clips = list(fits[name]["clips"].values())
        rates = nominal_rates(clips[0]["params"])
        bands = []
        for c in clips:
            level, _ = model_ltas(c["params"], rates)
            _, band = third_octave(LTAS_FREQS, level)
            bands.append(band)
        bands = np.asarray(bands)
        dev = bands - bands[0]
        shape = dev - dev.mean(axis=1, keepdims=True)
        ltas_dev.append(
            {
                "fit": name,
                "rms_db": [float(v) for v in np.sqrt((dev**2).mean(axis=1))],
                "shape_rms_db": [float(v) for v in np.sqrt((shape**2).mean(axis=1))],
                "band_max_db": [float(v) for v in np.abs(dev).max(axis=1)],
                "level_db": [float(v) for v in dev.mean(axis=1)],
            }
        )
    refit_dev = []
    for a, b in pairs:
        for cid in sorted(set(fits[a]["clips"]) & set(fits[b]["clips"])):
            pa = fits[a]["clips"][cid]["params"]
            pb = fits[b]["clips"][cid]["params"]
            rates = nominal_rates(pa)
            la, _ = model_ltas(pa, rates)
            lb, _ = model_ltas(pb, rates)
            _, ba = third_octave(LTAS_FREQS, la)
            _, bb = third_octave(LTAS_FREQS, lb)
            d = bb - ba
            refit_dev.append(
                {
                    "pair": f"{a}|{b}",
                    "clip": cid,
                    "rms_db": float(np.sqrt((d**2).mean())),
                    "shape_rms_db": float(np.sqrt(((d - d.mean()) ** 2).mean())),
                    "band_max_db": float(np.abs(d).max()),
                    "level_db": float(d.mean()),
                }
            )
            break  # one clip per pair is enough; the parameters are clip-tied
    ltas = {
        "per_fit": ltas_dev,
        "max_rms_db": float(max(max(d["rms_db"]) for d in ltas_dev)),
        "max_band_db": float(max(max(d["band_max_db"]) for d in ltas_dev)),
        "per_refit_pair": refit_dev,
        "max_rms_db_refit": float(max(d["rms_db"] for d in refit_dev)),
        "max_band_db_refit": float(max(d["band_max_db"] for d in refit_dev)),
        "note": (
            "model LTAS of every clip of a fit, all evaluated at the ANCHOR clip's "
            "rates so only the parameters differ; deviation from the anchor clip in "
            "1/3-octave bands 50 Hz - 7.9 kHz. The between-refit rows are the same "
            "measurement for two fits of the SAME clip, and they are SMALLER than the "
            "between-clip rows even though the individual parameters differ far more: "
            "the refits' parameter differences compensate, which is why the sampler's "
            "independent per-parameter draws need the stated factor on the envelope."
        ),
    }

    return {
        "provenance": provenance,
        "level_convention": {
            "power_scale_folded": True,
            "folded_via": "revised_eval._to_physical (ABSOLUTE_POWER_DB_FIELDS)",
            "rate_factor_applied": False,
            "rate_factor_reason": (
                "revised_eval.to_renderer_units' 10 log10(work/analysis) term (+4.4032 dB at "
                "the 44.1 kHz work grid) converts PERIODOGRAM units; this module's level "
                "coordinate is a whole-array RMS, which decimation preserves. Measured: with "
                "power_scale folded and normalize_rms=None the FLY125 cruise anchor renders at "
                "-24.048 dBFS against its real clip's -23.623 dBFS (0.43 dB); adding the term "
                "would put it 3.98 dB above the real clip"
            ),
        },
        "decomposition": {
            "form": "profile_db[r,k] = gain + slope*(log10 k - mean) + env(k)*s(k) + resid(k)",
            "parity_smooth_dec": PARITY_SMOOTH_DEC,
            "last_order_excluded_from_statistics": True,
            "compare": decomp_compare,
        },
        "per_profile": per_profile,
        "residual_correlation": resid_corr,
        "parity": parity,
        "trend": trend,
        "between_clip": between_clip,
        "between_refit": between_refit,
        "between_rig": between_rig,
        "ltas_between_clip": ltas,
        "widths_strength1": WIDTHS.as_dict(),
    }


# ---------------------------------------------------------------------------
# phase B: the widths, frozen from phase A
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Widths:
    """Per-draw sigmas at ``strength = 1``. Every field traces to phase A.

    ``provenance`` names the tier and the ``structure.json`` path each number
    comes from. ``main()`` prints frozen against freshly measured.
    """

    rotor_gain_db: float = 3.577
    slope_db_dec: float = 1.456
    parity_ln: float = 0.080
    resid_redraw_frac: float = 0.407
    resid_length_dec: float = 0.073
    gamma_ln: float = 0.045
    floor_mean_db: float = 1.066
    floor_tilt_db_oct: float = 0.210
    floor_shape_db: float = 2.175
    floor_shape_ar1: float = 0.838
    mic_gain_db: float = 1.122
    mic_floor_db: float = 0.199
    gain_all_db: float = 0.127
    coherence_ln: float = 0.198
    amp_exp: float = 2.733
    floor_exp: float = 2.733
    floor_static_rel_ln: float = 0.332

    def as_dict(self) -> dict[str, Any]:
        return {
            "sigma": {k: getattr(self, k) for k in self.__dataclass_fields__},
            "provenance": PROVENANCE,
            "guards": {
                "trend_margin_db": TREND_MARGIN_DB,
                "gamma_excursion_cap": GAMMA_EXCURSION_CAP,
                "real_pair_rms_db": REAL_PAIR_RMS_DB,
                "real_pair_band_db": REAL_PAIR_BAND_DB,
                "ltas_envelope_x": LTAS_ENVELOPE_X,
                "level_band_hz": list(LEVEL_BAND_HZ),
            },
        }


PROVENANCE: dict[str, str] = {
    "rotor_gain_db": "between_clip.rotor_gain_db.pooled_rms",
    "slope_db_dec": "between_refit.slope_db_dec.sigma (FALLBACK: tied between clips)",
    "parity_ln": "between_refit.parity_ln.sigma (FALLBACK: tied between clips)",
    "resid_redraw_frac": "between_refit.resid_redraw_frac.median (FALLBACK: tied)",
    "resid_length_dec": "residual_correlation.length_variogram_dec_median",
    "gamma_ln": "between_refit.gamma_ln.sigma (FALLBACK: tied between clips)",
    "floor_mean_db": "between_clip.floor_mean_db.pooled_rms",
    "floor_tilt_db_oct": "between_refit.floor_tilt_db_oct.sigma (FALLBACK: tied)",
    "floor_shape_db": "between_refit.floor_shape_db.sigma, mean-removed (FALLBACK: tied)",
    "floor_shape_ar1": "between_refit.floor_shape_db.lag1_corr_mean",
    "mic_gain_db": "between_refit.mic_gain_db.sigma (FALLBACK: tied between clips)",
    "mic_floor_db": "between_refit.mic_floor_db.sigma (FALLBACK: tied between clips)",
    "gain_all_db": "between_refit.gain_all_db.sigma (FALLBACK: tied between clips)",
    "coherence_ln": "between_clip.coherence_k_half_ln.pooled_rms",
    "amp_exp": "between_refit.amp_exp.sigma (FALLBACK: tied between clips)",
    "floor_exp": (
        "between_refit.amp_exp.sigma reused; floor_exp's own refit sigma is 3.53 and "
        "includes a +6.79 -> -2.71 sign flip between two fits of the same audio, an "
        "identifiability artefact rather than a neighbourhood"
    ),
    "floor_static_rel_ln": "between_refit.floor_static_rel_ln.sigma (FALLBACK: tied)",
}

WIDTHS = Widths()

#: the trend must fall by at least this much from k=1 to k=K, per rotor.
#: the smallest total trend drop over the 120 measured CRUISE rotor profiles is
#: 15.0 dB (``trend.drop_db_min_cruise``), so every measured cruise rotor
#: clears this with 12 dB to spare
TREND_MARGIN_DB = 3.0

#: linewidths may move a bit, never a lot: total excursion of gamma0 and
#: gamma_slope is capped at this factor. STATED, not measured: it keeps a
#: sampled linewidth inside the anchor's own decade, well short of the 3.6x
#: jump between the two DREGON fits and the 14.7x between-rig span
GAMMA_EXCURSION_CAP = 1.5

#: LTAS plausibility envelope, twice re-scoped and this is the reasoning chain.
#:
#: (1) It started as the BETWEEN-CLIP spread (3.88 dB rms on the physical
#: scale, still in ``structure.json:ltas_between_clip``). That encodes "is this
#: the SAME drone", which is the wrong question for a transfer arm whose job is
#: to COVER real rigs, and it rejected 43 % of strength-2 draws.
#: (2) It was then tied to the ANCHOR-TO-REAL deviation, which on the
#: confounded whole-array-RMS coordinate looked like 6.5-6.7 dB rms. On the
#: CORRECTED coordinate — declared rate term applied, restricted to
#: :data:`LEVEL_BAND_HZ` — the anchor-to-real deviation is only 2.09 dB rms
#: (FLY125 cruise) and 0.59 dB rms (DREGON room-2 cruise): above 300 Hz these
#: fits already agree with their own recordings. A stated multiple of THAT is
#: tighter than the between-clip spread and cannot coexist with a strength-2
#: arm (a strength-2 draw deviates 6.7 dB rms in that band), so it too is the
#: wrong basis: it measures fit quality, not rig-to-rig variability.
#: (3) The basis used now is the measured difference between TWO REAL DRONES of
#: this kind — the FLY125 cruise clip against the DREGON room-2 cruise clip,
#: both real audio, in :data:`LEVEL_BAND_HZ`: 4.97 dB rms, 7.31 dB worst band
#: (27.07 dB worst band full-band, dominated by the low-order region). It is
#: the only non-circular envelope available: it does not depend on the fits, on
#: the sampler's own widths, or on the strength requested.
REAL_PAIR_RMS_DB = 4.97
REAL_PAIR_BAND_DB = 7.31
#: ...times this STATED factor: a sampled rig may differ from its anchor by up
#: to twice as much as two real drones of this kind differ from each other.
#: NOT scaled by strength — an envelope that widened with the request could
#: never refuse one.
LTAS_ENVELOPE_X = 2.0
#: The guard constrains the RMS over bands only; the worst single band is
#: reported and NOT guarded. Reason: at these rotor rates a 1/3-octave band
#: between 300 and 700 Hz holds one or two rotor orders, so the per-order
#: fine-structure freedom the user asked for ("harmonic profiles may vary
#: QUITE FREELY") necessarily moves individual bands by 10-20 dB — measured
#: mean worst-band 12.4 dB at strength 2 against a real-pair worst band of
#: 7.31 dB. Guarding the worst band would guard exactly what was asked to vary.
#: The structural guards — parity sign, falling power-law trend, gamma
#: excursion, finite render, ``floor_exp >= 0`` — are NOT re-scoped. They
#: encode "is this a drone at all", which is the user's invariant and does not
#: depend on how far the draw has travelled.


class SampleRejected(RuntimeError):
    """No draw satisfied the guards within the attempt budget."""


# ---------------------------------------------------------------------------
# phase B: the sampler
# ---------------------------------------------------------------------------


def _sqexp_draw(rng: np.random.Generator, x: np.ndarray, length: float) -> np.ndarray:
    """One unit-variance squared-exponential draw on the ``log k`` axis.

    PROJECTED off the trend basis ``{1, x}`` and rescaled to unit variance.
    The residual coordinate is by construction what is left after the
    least-squares trend fit, i.e. orthogonal to ``{1, x}``; a raw draw is not,
    and its projection would leak into the gain and slope coordinates. It
    measurably does: before the projection the realised slope spread over the
    demo's draws was 1.86 dB/decade against the 1.46 drawn, the excess being
    the draw's own tilt.
    """
    d = x[:, None] - x[None, :]
    cov = np.exp(-0.5 * (d / length) ** 2) + 1e-8 * np.eye(x.size)
    chol = np.linalg.cholesky(cov)
    z = chol @ rng.standard_normal(x.size)
    design = np.stack([np.ones(x.size), x - x.mean()], axis=1)
    coef, *_ = np.linalg.lstsq(design, z, rcond=None)
    z = z - design @ coef
    return z / max(float(z.std()), 1e-12)


def _ar1_draw(rng: np.random.Generator, n: int, rho: float) -> np.ndarray:
    """Unit-variance AR(1) sequence with lag-1 correlation ``rho``."""
    out = np.empty(n)
    out[0] = rng.standard_normal()
    for i in range(1, n):
        out[i] = rho * out[i - 1] + np.sqrt(max(1.0 - rho * rho, 0.0)) * rng.standard_normal()
    return out


def _rebase_divergent_floor(
    export: dict[str, Any], speed_rps: float | np.ndarray
) -> tuple[dict[str, Any], str]:
    """Set ``floor_exp = 0`` level-preservingly when it is negative.

    ``build_psd`` scales the rotors' share of the floor by
    ``mean_r (s_r / 80)**floor_exp + floor_static_rel``, so a negative exponent
    makes the floor DIVERGE as the rotors stop and a telemetry sample of
    exactly zero rev/s renders ``inf``. Folding the anchor's own floor gain at
    its own speed into ``floor_mean_db`` leaves the floor unchanged at
    ``speed_rps`` and simply stops it extrapolating below. Same two lines as
    ``flight_model._rebase_divergent_floor``, reimplemented here because that
    module is owned elsewhere.

    ``speed_rps`` may be the ``(R,)`` per-rotor rates rather than one anchor
    speed. They are reduced to the ROTOR-POWER-MEAN equivalent speed
    ``80 * (mean_r (s_r/80)**floor_exp)**(1/floor_exp)``, which is the speed
    whose gain equals the per-rotor mean ``build_psd`` actually forms, so the
    rebase is exactly level-preserving on a rig whose rotors run at different
    rates. ``flight_model`` passes a single anchor speed because it has one.
    """
    floor_exp = float(export.get("floor_exp", export.get("amp_exp", 0.0)))
    if floor_exp >= 0.0:
        return export, ""
    s = np.atleast_1d(np.asarray(speed_rps, dtype=np.float64))
    speed_rps = float(
        AMP_RPS_REF * np.mean((np.maximum(s, 1e-9) / AMP_RPS_REF) ** floor_exp) ** (1.0 / floor_exp)
    )
    rel = max(float(export.get("floor_static_rel", 0.0)), 0.0)
    gain_at_anchor = (max(speed_rps, 1e-9) / AMP_RPS_REF) ** floor_exp + rel
    shift_db = 10.0 * np.log10(gain_at_anchor / (1.0 + rel))
    out = dict(export)
    out["floor_exp"] = 0.0
    out["floor_mean_db"] = float(export.get("floor_mean_db", 0.0)) + float(shift_db)
    return out, (
        f"floor_exp {floor_exp:+.3f} clamped to 0.0 with floor_mean_db {shift_db:+.3f} dB "
        f"so the floor is unchanged at {speed_rps:.2f} rev/s"
    )


def check_sample(
    export: dict[str, Any],
    reference: dict[str, Any],
    *,
    ltas_tol: float | None = None,
) -> dict[str, Any]:
    """Guard results for one sampled export against its anchor.

    Returns a dict with ``ok`` and ``failed`` plus every measured quantity, so
    a caller can count which guard fired how often. Uses the model LTAS (the
    render's expectation) rather than a render, which is what makes it cheap
    enough to run inside :func:`sample_rig`'s redraw loop; the finiteness guard
    is evaluated at the anchor speed AND at exactly zero speed, where a
    negative ``floor_exp`` diverges.

    ``ltas_tol`` overrides the LTAS plausibility envelope: the maximum RMS
    1/3-octave deviation, in dB, inside :data:`LEVEL_BAND_HZ`. Default
    ``LTAS_ENVELOPE_X * REAL_PAIR_RMS_DB``.
    """
    rates = nominal_rates(reference)
    out: dict[str, Any] = {}

    level, total = model_ltas(export, rates)
    finite_anchor = bool(np.all(np.isfinite(total)) and np.all(total > 0.0))
    _, total0 = model_ltas(export, np.zeros_like(rates))
    finite_zero = bool(np.all(np.isfinite(total0)))
    out["finite_at_anchor_speed"] = finite_anchor
    out["finite_at_zero_speed"] = finite_zero
    out["finite"] = finite_anchor and finite_zero
    # The amplitude law is (rps / 80)**amp_exp, so a NEGATIVE exponent is
    # 0**negative = inf on a trajectory that touches EXACTLY zero rotor speed —
    # which every full-flight training window does, its ground phase being
    # exact zeros — and fm_lines turns that into NaN audio (verified: a
    # half-stopped trajectory with amp_exp = -7.27 renders max|x| = nan). The
    # zero-speed PSD probe above does NOT catch it, because at zero speed every
    # line of a stopped rotor collapses onto DC and the infinity does not
    # survive the scatter. It is the same invariant as floor_exp >= 0, but
    # amp_exp is GUARDED rather than clamped: clamping it to 0 would make the
    # comb speed-INDEPENDENT, i.e. as loud at idle as at cruise, which is a
    # worse statement than redrawing. Reported by the stream-wiring side, which
    # hit it on the Michael<->DREGON path at spread 2-3.
    out["amp_exp"] = float(export.get("amp_exp", 0.0))
    out["floor_exp"] = float(export.get("floor_exp", 0.0))
    out["speed_law"] = bool(out["amp_exp"] >= 0.0 and out["floor_exp"] >= 0.0)

    prof = effective_profile(export)
    drops, parity = [], []
    for r in range(prof.shape[0]):
        parts = decompose_profile(prof[r])
        drops.append(parts.trend_drop_db)
        parity.append(parts.parity_low_db)
    out["trend_drop_db"] = drops
    out["trend_falls"] = bool(min(drops) >= TREND_MARGIN_DB)
    out["parity_low_db"] = parity
    out["parity_sign"] = bool(min(parity) > 0.0)

    ratio = []
    for key in ("gamma0", "gamma_slope"):
        a = float(np.mean(np.asarray(export[key], dtype=np.float64)))
        b = float(np.mean(np.asarray(reference[key], dtype=np.float64)))
        ratio.append(a / b if b > 0.0 else 1.0)
    out["gamma_ratio"] = ratio
    out["gamma_excursion"] = bool(
        all(1.0 / GAMMA_EXCURSION_CAP <= v <= GAMMA_EXCURSION_CAP for v in ratio)
    )

    ref_level, _ = model_ltas(reference, rates)
    cen, band = third_octave(LTAS_FREQS, level)
    _, ref_band = third_octave(LTAS_FREQS, ref_level)
    # restricted to the level band for the same reason the level coordinate is:
    # below 300 Hz the render carries a known k=1-3 excess that is present in
    # the anchor and in every sample alike, so it contributes nothing but noise
    # to a sample-vs-anchor comparison while inflating the envelope this guard
    # is calibrated on
    keep = (cen >= LEVEL_BAND_HZ[0]) & (cen <= LEVEL_BAND_HZ[1])
    dev = (band - ref_band)[keep]
    full_dev = band - ref_band
    shape = dev - dev.mean()
    out["ltas_rms_db"] = float(np.sqrt((dev**2).mean()))
    out["ltas_shape_rms_db"] = float(np.sqrt((shape**2).mean()))
    out["ltas_band_max_db"] = float(np.abs(dev).max())
    out["ltas_level_db"] = float(dev.mean())
    out["ltas_rms_db_full_band"] = float(np.sqrt((full_dev**2).mean()))
    out["ltas_band_max_db_full_band"] = float(np.abs(full_dev).max())
    tol = float(LTAS_ENVELOPE_X * REAL_PAIR_RMS_DB if ltas_tol is None else ltas_tol)
    out["ltas_tol_rms_db"] = tol
    out["ltas"] = bool(np.isfinite(out["ltas_rms_db"]) and out["ltas_rms_db"] <= tol)

    guards = ("finite", "speed_law", "trend_falls", "parity_sign", "gamma_excursion", "ltas")
    out["failed"] = [g for g in guards if not out[g]]
    out["ok"] = not out["failed"]
    return out


def _draw(
    export: dict[str, Any],
    rng: np.random.Generator,
    strength: float,
    widths: Widths,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One unguarded draw. Returns the new export and what was drawn."""
    s = float(strength)
    out = dict(export)
    rates = nominal_rates(export)
    prof = effective_profile(export)
    stored = np.atleast_2d(np.asarray(export["profile_db"], dtype=np.float64))
    n_rotors, n_orders = prof.shape

    # --- harmonic profile: gain, trend, parity, correlated residual --------
    d_gain = rng.normal(0.0, s * widths.rotor_gain_db, size=n_rotors)
    d_slope = rng.normal(0.0, s * widths.slope_db_dec, size=n_rotors)
    par_scale = np.exp(rng.normal(0.0, s * widths.parity_ln, size=n_rotors))
    # residual MIXING, not addition: resid' = rho resid + sqrt(1-rho^2) sigma z
    # keeps the fine-structure variance at the anchor's own measured level for
    # every strength, and saturates at a fully independent draw instead of
    # diverging. r = s * measured refit ratio, rho = 1 - r^2/2.
    r_mix = s * widths.resid_redraw_frac
    rho = float(np.clip(1.0 - 0.5 * r_mix * r_mix, 0.0, 1.0))
    new_profile = np.array(stored, dtype=np.float64)
    drawn_resid_rms = []
    for i in range(n_rotors):
        parts = decompose_profile(prof[i])
        z = _sqexp_draw(rng, parts.x, widths.resid_length_dec)
        fresh = rho * parts.resid + np.sqrt(max(1.0 - rho * rho, 0.0)) * parts.resid_std * z
        delta = (
            d_gain[i]
            + d_slope[i] * (parts.x - parts.x.mean())
            + (par_scale[i] - 1.0) * parts.env * parts.s
            + (fresh - parts.resid)
        )
        drawn_resid_rms.append(float(np.sqrt(np.mean((fresh - parts.resid) ** 2))))
        new_profile[i, :n_orders] = stored[i, :n_orders] + delta
    out["profile_db"] = new_profile.tolist()

    # --- linewidths: multiplicative only, and only a bit -------------------
    gamma_fac = float(np.exp(rng.normal(0.0, s * widths.gamma_ln)))
    for key in ("gamma0", "gamma_slope"):
        out[key] = (np.asarray(export[key], dtype=np.float64) * gamma_fac).tolist()
    # a numerically zero gamma0 (FLY125: 2e-5 bins) stays zero under a
    # multiplicative draw, which is the intended statement: that rig's width
    # is carried entirely by gamma_slope, and it is nonzero in every fit
    if "gamma" in out:
        out.pop("gamma")  # derived gamma0 + gamma_slope * k; stale after a draw

    # --- broadband floor ---------------------------------------------------
    shape = np.asarray(export["floor_shape_db"], dtype=np.float64)
    d_shape = _ar1_draw(rng, shape.size, widths.floor_shape_ar1) * s * widths.floor_shape_db
    out["floor_shape_db"] = (shape + (d_shape - d_shape.mean())).tolist()
    out["floor_mean_db"] = float(
        float(export["floor_mean_db"]) + rng.normal(0.0, s * widths.floor_mean_db)
    )
    out["floor_tilt_db_oct"] = float(
        float(export["floor_tilt_db_oct"]) + rng.normal(0.0, s * widths.floor_tilt_db_oct)
    )
    out["floor_static_rel"] = float(
        float(export.get("floor_static_rel", 0.0))
        * float(np.exp(rng.normal(0.0, s * widths.floor_static_rel_ln)))
    )

    # --- channel structure -------------------------------------------------
    for key, sigma in (
        ("mic_gain_db", widths.mic_gain_db),
        ("mic_floor_db", widths.mic_floor_db),
        ("gain_all_db", widths.gain_all_db),
    ):
        base = np.asarray(export[key], dtype=np.float64)
        out[key] = (base + rng.normal(0.0, s * sigma, size=base.shape)).tolist()

    # --- coherence and the speed law --------------------------------------
    out["coherence_k_half"] = float(
        float(export["coherence_k_half"]) * float(np.exp(rng.normal(0.0, s * widths.coherence_ln)))
    )
    out["amp_exp"] = float(float(export["amp_exp"]) + rng.normal(0.0, s * widths.amp_exp))
    out["floor_exp"] = float(
        float(export.get("floor_exp", export["amp_exp"])) + rng.normal(0.0, s * widths.floor_exp)
    )
    out, rebase = _rebase_divergent_floor(out, rates)

    drawn = {
        "d_gain_db": [float(v) for v in d_gain],
        "d_slope_db_dec": [float(v) for v in d_slope],
        "parity_scale": [float(v) for v in par_scale],
        "resid_mix_rho": rho,
        "resid_perturb_rms_db": drawn_resid_rms,
        "gamma_factor": gamma_fac,
        "floor_exp_rebase": rebase,
        "amp_exp": out["amp_exp"],
        "floor_exp": out["floor_exp"],
        "coherence_k_half": out["coherence_k_half"],
        "floor_mean_db": out["floor_mean_db"],
    }
    return out, drawn


def sample_rig(
    export: dict[str, Any],
    rng: np.random.Generator,
    *,
    strength: float = 1.0,
    max_attempts: int = 16,
    widths: Widths = WIDTHS,
    ltas_tol: float | None = None,
) -> dict[str, Any]:
    """One perturbed export: a plausible other drone of the same kind.

    ``strength = 1`` is the measured between-clip neighbourhood for the three
    quantities that vary between clips and the between-refit neighbourhood for
    every quantity the stage-2 fit ties (see the module docstring).
    ``strength > 1`` scales every perturbation.

    Guards are applied by :func:`check_sample` and a failing draw is REDRAWN,
    never clipped, so the accepted sample is an honest draw from the truncated
    distribution. Raises :class:`SampleRejected` after ``max_attempts``.

    The returned dict is a new export of the same schema with JSON-native
    values; untouched fields are passed through from ``export``. The derived
    ``gamma`` field is dropped, since it is ``gamma0 + gamma_slope * k`` and the
    renderer rebuilds it.
    """
    attempts: list[dict[str, Any]] = []
    for _ in range(int(max_attempts)):
        cand, drawn = _draw(export, rng, strength, widths)
        guards = check_sample(cand, export, ltas_tol=ltas_tol)
        attempts.append(guards)
        if guards["ok"]:
            cand["_sampler"] = {
                "mode": "neighbourhood",
                "strength": float(strength),
                "attempts": len(attempts),
                "rejected": [a["failed"] for a in attempts[:-1]],
                "drawn": drawn,
                "guards": guards,
            }
            return cand
    counts: dict[str, int] = {}
    for a in attempts:
        for g in a["failed"]:
            counts[g] = counts.get(g, 0) + 1
    raise SampleRejected(
        f"no draw satisfied the guards in {max_attempts} attempts at strength "
        f"{strength:g}; guards fired: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )


def sample_batch(
    export: dict[str, Any],
    n: int,
    seed: int,
    *,
    strength: float = 1.0,
    max_attempts: int = 16,
    widths: Widths = WIDTHS,
    ltas_tol: float | None = None,
) -> list[dict[str, Any]]:
    """``n`` samples, each from its own substream of ``seed`` so any one is
    reproducible on its own."""
    out = []
    for i in range(int(n)):
        rng = np.random.default_rng([int(seed), i])
        out.append(
            sample_rig(
                export,
                rng,
                strength=strength,
                max_attempts=max_attempts,
                widths=widths,
                ltas_tol=ltas_tol,
            )
        )
    return out


# ---------------------------------------------------------------------------
# phase B, mode 2: the path cloud between two anchors
# ---------------------------------------------------------------------------

#: the ONLY fields ``stage2.params_from_export`` reads. Everything else in an
#: export is per-clip bookkeeping (``knots_s``, ``floor_level_db``,
#: ``floor_tilt_gp``, ``rps_offset``, the derived ``gamma``) and is inert for
#: :func:`stage2.render_from_export`.
RENDERABLE_FIELDS = (
    "profile_db",
    "gamma0",
    "gamma_slope",
    "floor_ctrl_hz",
    "floor_shape_db",
    "floor_tilt_db_oct",
    "floor_mean_db",
    "floor_static_rel",
    "amp_exp",
    "floor_exp",
    "mic_gain_db",
    "mic_floor_db",
    "gain_all_db",
    "coherence_k_half",
)

#: interpolation convention per path coordinate, quoted in the demo's index
PATH_INTERP: dict[str, str] = {
    "rotor_gain_db": "linear in dB (geometric in power)",
    "trend_slope_db_dec": "linear (already a log-log slope)",
    "parity_envelope_db": "linear in dB, elementwise on the shared order grid",
    "residual_db": "linear in dB, elementwise on the shared order grid",
    "gamma0": (
        "linear in bins, NOT log: FLY125's gamma0 is numerically zero (2e-5) and a "
        "geometric path from zero is degenerate"
    ),
    "gamma_slope": "log-linear (strictly positive in every fit, spans 14.7x)",
    "coherence_k_half": "log-linear (strictly positive scale)",
    "floor_mean_db": "linear in dB",
    "floor_tilt_db_oct": "linear in dB/octave",
    "floor_shape_db": (
        "linear in dB per control point; if the two fits' floor_ctrl_hz grids differ, "
        "B's shape is resampled onto A's grid linearly in log frequency"
    ),
    "floor_static_rel": "linear in the power ratio itself (it is a power share, and 0 is legal)",
    "amp_exp": "linear",
    "floor_exp": "linear, then hard-clamped to >= 0 level-preservingly",
    "mic_gain_db": "linear in dB",
    "mic_floor_db": "linear in dB",
    "gain_all_db": "linear in dB, after the overall level has been removed from it",
    "carrier": "linear in rev/s (the hybrid rig's operating point; inert for rendering)",
    "overall_level_db": ("linear in dBFS, carried as ONE scalar in gain_all_db; see level_free()"),
}


#: The level coordinate's band, in Hz. NOT the full band: with the declared
#: rate term applied, both rigs' renders agree with their real clips to within
#: ~1.8 dB in every band above 300 Hz with no gain fitted at all, while
#: 50-200 Hz carries a large excess concentrated in rotor orders k=1-3 (DREGON
#: k=1 is +26.4 dB, k=2 +18.6 dB; the floor BETWEEN the lines is right to about
#: 1 dB at every probed order) that is a localised render-path defect under
#: investigation, not a level. A whole-array RMS is power-weighted, so those
#: few low orders dominate it and it is unusable as a level coordinate — and it
#: would stay unusable for any rig whose profile or gamma puts different weight
#: at low order. Everything this module calls a "level" is therefore the power
#: in 300 Hz - 7.9 kHz.
LEVEL_BAND_HZ = (300.0, 7900.0)


def band_level_db(x: np.ndarray, band: tuple[float, float] = LEVEL_BAND_HZ) -> float:
    """Power of a real or rendered signal inside ``band``, in dB, mic mean."""
    cen, levels = audio_bands(x)
    m = (cen >= band[0]) & (cen <= band[1])
    return float(10.0 * np.log10(max(float(np.mean(10.0 ** (levels[m] / 10.0))), 1e-300)))


def render_level_db(
    export: dict[str, Any],
    rps: np.ndarray,
    *,
    seed: int = 0,
    sample_rate_work: int = RENDER_SAMPLE_RATE_WORK,
    n_mics: int = 8,
    band: tuple[float, float] = LEVEL_BAND_HZ,
) -> float:
    """This export's own render level in :data:`LEVEL_BAND_HZ`, dB.

    The path sampler's level coordinate: an export shifted by
    ``-render_level_db`` renders at unit band power at its own anchor speed on
    its own telemetry. ``normalize_rms=None``, and the export must already be
    on the physical scale (:func:`to_physical`) for the number to mean
    anything. Band-restricted for the reason given at :data:`LEVEL_BAND_HZ`;
    an earlier revision used the whole-array RMS and that was wrong.
    """
    from experiments.stochastic_fit import stage2

    audio = stage2.render_from_export(
        export,
        rps,
        sample_rate_work=sample_rate_work,
        n_mics=n_mics,
        seed=seed,
        normalize_rms=None,
    )
    return band_level_db(audio, band)


def model_level_db(export: dict[str, Any], rates: np.ndarray | None = None) -> float:
    """In-band model power in dB, microphone structure included.

    The telemetry-free fallback for :func:`render_level_db`, and NOT
    interchangeable with it. MEASURED on the two demo anchors: the rendered
    offset FLY125 cruise -> DREGON room-2 cruise is +10.57 dB while this
    integral gives +2.59 dB, an 8.0 dB rig-dependent disagreement. The cause is
    in the renderer, not here: in ``line_mode="fm"``
    :func:`~data_processing.stochastic_rotor_noise.synthesize` sets the line
    level from the ratio of MEAN line spectrum to mean floor spectrum over the
    WHOLE 0-22.05 kHz work grid, against the realised floor variance, and only
    then antialiases to the 8 kHz output band — so the in-band line/floor
    balance of a render depends on out-of-band content and does not equal this
    in-band integral. Use :func:`render_level_db` whenever a clip exists; use
    this only when none does, and do not compare its value across rigs without
    saying so.
    """
    from experiments.stochastic_fit import stage2

    if rates is None:
        rates = nominal_rates(export)
    rates = np.asarray(rates, dtype=np.float64)
    params = stage2.params_from_export(export, rates, sample_rate=32000, n_mics=8)
    from data_processing import stochastic_rotor_noise as srn

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        psd = srn.build_psd(
            params, rates[:, None], LTAS_FREQS, dt=0.1, rng=np.random.default_rng(0)
        )
    floor = psd["floor"][0]  # (F,)
    lines = psd["lines"][:, 0, :]  # (R, F)
    mic_gain = 10.0 ** (np.asarray(export["mic_gain_db"], dtype=np.float64) / 10.0)
    mic_floor = 10.0 ** (np.asarray(export["mic_floor_db"], dtype=np.float64) / 10.0)
    # gain_all_db is applied to the AUDIO as 10**(g/20), i.e. one dB of
    # gain_all_db is one dB of level; in power that is 10**(g/10)
    gain_all = 10.0 ** (np.asarray(export["gain_all_db"], dtype=np.float64) / 10.0)
    per_mic = (mic_floor[:, None] * floor[None, :] + mic_gain @ lines) * gain_all[:, None]
    df = float(LTAS_FREQS[1] - LTAS_FREQS[0])
    return float(10.0 * np.log10(max(float(per_mic.sum(axis=1).mean()) * df, 1e-300)))


def level_free(export: dict[str, Any], level_db: float) -> dict[str, Any]:
    """``export`` with its overall level removed into the level coordinate.

    The level is carried by ``gain_all_db``, which ``synthesize`` applies to
    the finished audio as ``10**(gain_all_db / 20)`` — one dB of it is one dB
    of output level, and it multiplies lines and floor alike, so it is the
    exact carrier of an overall gain. The SAME scalar is subtracted from all
    eight entries, so the rig's channel pattern is untouched.
    """
    out = dict(export)
    out["gain_all_db"] = (
        np.asarray(export["gain_all_db"], dtype=np.float64) - float(level_db)
    ).tolist()
    return out


def interpolate_exports(
    export_a: dict[str, Any],
    export_b: dict[str, Any],
    t: float,
    *,
    levels: tuple[float, float],
    level_db: float | None = None,
) -> dict[str, Any]:
    """The point at mixing coordinate ``t`` on the path from A to B.

    Interpolation happens in the DECOMPOSED profile coordinates (per-rotor
    gain, trend slope in dB/decade, parity envelope, residual) and in the
    scalar rig coordinates, one convention per quantity, listed in
    :data:`PATH_INTERP`. It does NOT interpolate raw ``profile_db`` arrays
    elementwise across fits of different length: the shared order grid is
    ``K = min(K_a, K_b)`` and the extra orders of the longer fit are DROPPED,
    because padding a profile past its own fit's order support would invent
    data. For the intended Michael(111)/DREGON(109) pair that drops 2 orders,
    above 7.74 kHz.

    ``levels`` are the two anchors' overall levels in dBFS (see
    :func:`render_level_db`). They are removed before interpolating anything
    level-like and the interpolated level is added back as one scalar, so the
    shape coordinates are mixed on a common scale instead of inheriting two
    incompatible ``normalize_rms=0.1`` gain conventions. ``level_db`` overrides
    the interpolated level, which is how a caller pins a whole cloud to one
    playback level. Note honestly: with the default dB-linear level convention
    the reconstruction is ALGEBRAICALLY identical to interpolating the raw
    level fields — every step is linear in dB. What the split buys is that the
    offset between the anchors is measured and recorded rather than silently
    mixed, and that ``level_db`` exists at all.

    Both parity sign and a falling trend survive this interpolation by
    construction: a convex combination of two positive parity envelopes is
    positive and a convex combination of two negative trend slopes is negative,
    so the interpolated point always satisfies the structural invariants and
    only the perturbation on top of it can break them.
    """
    t = float(t)
    la, lb = float(levels[0]), float(levels[1])
    a = level_free(export_a, la)
    b = level_free(export_b, lb)
    lvl = (1.0 - t) * la + t * lb if level_db is None else float(level_db)

    prof_a, prof_b = effective_profile(a), effective_profile(b)
    n_rotors = min(prof_a.shape[0], prof_b.shape[0])
    n_orders = min(prof_a.shape[1], prof_b.shape[1])
    profile = np.empty((n_rotors, n_orders))
    coords: dict[str, list[float]] = {"gain_db": [], "slope_db_dec": [], "parity_low_db": []}
    for r in range(n_rotors):
        pa = decompose_profile(prof_a[r, :n_orders])
        pb = decompose_profile(prof_b[r, :n_orders])
        gain = (1.0 - t) * pa.gain + t * pb.gain
        slope = (1.0 - t) * pa.slope + t * pb.slope
        env = (1.0 - t) * pa.env + t * pb.env
        resid = (1.0 - t) * pa.resid + t * pb.resid
        profile[r] = gain + slope * (pa.x - pa.x.mean()) + env * pa.s + resid
        coords["gain_db"].append(float(gain))
        coords["slope_db_dec"].append(float(slope))
        coords["parity_low_db"].append(float(env[:PARITY_LOW_ORDERS].mean()))

    def lin(key: str) -> np.ndarray:
        va = np.asarray(a[key], dtype=np.float64)
        vb = np.asarray(b[key], dtype=np.float64)
        return (1.0 - t) * va + t * vb

    def log_lin(key: str) -> float:
        va = float(np.mean(np.asarray(a[key], dtype=np.float64)))
        vb = float(np.mean(np.asarray(b[key], dtype=np.float64)))
        if min(va, vb) <= 0.0:
            return (1.0 - t) * va + t * vb
        return float(np.exp((1.0 - t) * np.log(va) + t * np.log(vb)))

    out = dict(a)
    out["profile_db"] = profile.tolist()
    out.pop("h_db", None)  # its time-mean is folded into profile_db above
    out.pop("gamma", None)  # derived from gamma0 + gamma_slope * k
    out["gamma0"] = np.resize(lin("gamma0"), n_rotors).tolist()
    out["gamma_slope"] = (np.ones(n_rotors) * log_lin("gamma_slope")).tolist()
    out["coherence_k_half"] = log_lin("coherence_k_half")
    ctrl_a = np.asarray(a["floor_ctrl_hz"], dtype=np.float64)
    ctrl_b = np.asarray(b["floor_ctrl_hz"], dtype=np.float64)
    shape_b = np.asarray(b["floor_shape_db"], dtype=np.float64)
    if ctrl_a.shape != ctrl_b.shape or not np.allclose(ctrl_a, ctrl_b):
        shape_b = np.interp(np.log(ctrl_a), np.log(ctrl_b), shape_b)
    out["floor_shape_db"] = (
        (1.0 - t) * np.asarray(a["floor_shape_db"], dtype=np.float64) + t * shape_b
    ).tolist()
    out["floor_ctrl_hz"] = ctrl_a.tolist()
    for key in ("floor_mean_db", "floor_tilt_db_oct", "amp_exp", "floor_exp", "floor_static_rel"):
        out[key] = float(np.mean(lin(key)))
    for key in ("mic_gain_db", "mic_floor_db"):
        out[key] = lin(key).tolist()
    out["gain_all_db"] = (lin("gain_all_db") + lvl).tolist()
    car_a = np.asarray(a["carrier"], dtype=np.float64)
    car_b = np.asarray(b["carrier"], dtype=np.float64)
    out["carrier"] = (
        ((1.0 - t) * car_a + t * car_b).tolist() if car_a.shape == car_b.shape else car_a.tolist()
    )
    # the interpolated point must itself be a legal, finite-rendering export:
    # floor_exp crosses zero on the FLY125(+6.79) -> DREGON(-3.71) path at
    # t = 0.646, and beyond that the floor diverges at zero speed
    out, rebase = _rebase_divergent_floor(out, nominal_rates(out))
    out["_path"] = {
        "t": t,
        "level_db": lvl,
        "levels": [la, lb],
        "level_offset_removed_db": lb - la,
        "n_orders": n_orders,
        "orders_dropped": [int(prof_a.shape[1] - n_orders), int(prof_b.shape[1] - n_orders)],
        "coords": coords,
        "carrier_shared_grid": bool(car_a.shape == car_b.shape),
        "floor_exp_rebase": rebase,
    }
    return out


def sample_path(
    export_a: dict[str, Any],
    export_b: dict[str, Any],
    rng: np.random.Generator,
    *,
    t: float | None = None,
    spread: float = 1.0,
    levels: tuple[float, float] | None = None,
    level_db: float | None = None,
    max_attempts: int = 16,
    widths: Widths = WIDTHS,
    ltas_tol: float | None = None,
) -> dict[str, Any]:
    """A draw from the CLOUD along the path between two anchor fits.

    ``t`` is the mixing coordinate, uniform on ``[0, 1]`` when ``None``; ``t=0``
    is anchor A and ``t=1`` is anchor B. ``spread`` is the neighbourhood width
    applied AROUND the interpolated point, in the same units as
    :func:`sample_rig`'s ``strength``, so ``spread=0`` returns the interpolated
    point itself and the endpoints reproduce the anchors' renderable
    parameters (exactly, except for the level-preserving ``floor_exp`` clamp —
    use :func:`renderable_deviation` to see both).

    ``levels`` are the anchors' overall levels in dBFS; when omitted they are
    taken from :func:`model_level_db`, which needs no telemetry. Pass
    :func:`render_level_db` values when a clip is available. The guards are
    evaluated against the INTERPOLATED point, not against either anchor, which
    is what lets the cloud reach a DREGON-like rig from a Michael anchor while
    still refusing a corrupted one.
    """
    tt = float(rng.uniform()) if t is None else float(t)
    if levels is None:
        levels = (model_level_db(export_a), model_level_db(export_b))
    mid = interpolate_exports(export_a, export_b, tt, levels=levels, level_db=level_db)
    path = mid.pop("_path")
    attempts: list[dict[str, Any]] = []
    for _ in range(int(max_attempts)):
        cand, drawn = _draw(mid, rng, spread, widths)
        guards = check_sample(cand, mid, ltas_tol=ltas_tol)
        attempts.append(guards)
        if guards["ok"]:
            cand["_sampler"] = {
                "mode": "path",
                "strength": float(spread),
                "attempts": len(attempts),
                "rejected": [a["failed"] for a in attempts[:-1]],
                "drawn": drawn,
                "guards": guards,
                "path": path,
            }
            return cand
    counts: dict[str, int] = {}
    for at in attempts:
        for g in at["failed"]:
            counts[g] = counts.get(g, 0) + 1
    raise SampleRejected(
        f"no draw satisfied the guards in {max_attempts} attempts at t={tt:.3f}, spread "
        f"{spread:g}; guards fired: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )


def renderable_deviation(export: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """How far ``export`` is from ``reference`` in the fields the renderer reads.

    Per-field max absolute difference over the shared grid, plus the floor's
    level at the anchor speed (the quantity the ``floor_exp`` clamp preserves,
    so it is the meaningful comparison when the clamp has fired) and the model
    LTAS deviation. This is the endpoint check for :func:`sample_path`.
    """
    rates = nominal_rates(reference)
    out: dict[str, Any] = {"field_max_abs": {}}
    for key in RENDERABLE_FIELDS:
        if key in ("floor_exp", "floor_mean_db"):
            continue
        va = np.atleast_1d(np.asarray(export[key], dtype=np.float64))
        vb = np.atleast_1d(np.asarray(reference[key], dtype=np.float64))
        sl = tuple(slice(0, min(x, y)) for x, y in zip(va.shape, vb.shape))
        out["field_max_abs"][key] = float(np.abs(va[sl] - vb[sl]).max())
        if va.shape != vb.shape:
            out["field_max_abs"][key + "_shape"] = f"{va.shape} vs {vb.shape}"

    def floor_level(e: dict[str, Any]) -> float:
        exp = float(e.get("floor_exp", e["amp_exp"]))
        rel = max(float(e.get("floor_static_rel", 0.0)), 0.0)
        gain = float(np.mean((np.maximum(rates, 1e-9) / AMP_RPS_REF) ** exp)) + rel
        return float(e["floor_mean_db"]) + 10.0 * np.log10(max(gain, 1e-300))

    out["floor_anchor_level_db"] = float(floor_level(export) - floor_level(reference))
    out["floor_exp"] = [float(export.get("floor_exp", 0.0)), float(reference.get("floor_exp", 0.0))]
    la, _ = model_ltas(export, rates)
    lb, _ = model_ltas(reference, rates)
    _, ba = third_octave(LTAS_FREQS, la)
    _, bb = third_octave(LTAS_FREQS, lb)
    out["ltas_band_max_db"] = float(np.abs(ba - bb).max())
    out["ltas_rms_db"] = float(np.sqrt(((ba - bb) ** 2).mean()))
    return out


# ---------------------------------------------------------------------------
# phase B: what ONE bank entry is made of
# ---------------------------------------------------------------------------

#: The rate a bank entry is built at, i.e. the training stream's own
#: ``sample_rate``. An entry's must match its pool's, and the pool refuses a
#: bank loudly when it does not. Not to be confused with
#: :data:`RENDER_SAMPLE_RATE_WORK`, which is the 44.1 kHz work grid the level
#: conversion of :func:`to_physical` is declared against.
BANK_WORK_RATE = 16000

#: Microphones a bank entry carries. The fitted per-(mic, rotor) pattern is
#: copied into the entry, so this must match the policy's ``n_mics``.
BANK_N_MICS = 8

#: HWHM of a unit-std Gaussian, i.e. the gamma-to-shaft-jitter conversion.
GAUSS_HWHM = float(np.sqrt(2.0 * np.log(2.0)))

#: STATED BOUND on the speed exponents, and the only place in this pipeline
#: where a sampled value is bounded by anything other than a measured width.
#:
#: WHY. The sampler draws ``amp_exp`` and ``floor_exp`` with the measured
#: between-refit sigma of 2.733, which is a legitimate WIDTH but has a tail no
#: fit supports: an unbounded bank reached 24.7 and 31.9 dB per dB of rotor
#: speed, and at that exponent an idle or ramp window carries a comb ~20 dB
#: less prominent than the same rig at cruise — the low-speed part of every
#: training clip would be a fiction. The bound is the range the six measured
#: fits actually occupy (``results/rig_sampler/structure.json``,
#: ``between_rig.per_fit``): ``amp_exp`` 4.398 (michael_standby) to 14.111
#: (dregon_flight), ``floor_exp`` -3.711 (dregon_cruise_refined) to 6.792
#: (michael_cruise).
#:
#: The ``floor_exp`` floor is raised from the measured -3.711 to 0 for a reason
#: that is not statistical: a negative exponent is ``0 ** negative`` at the
#: exact zero of a full flight's ground phase, i.e. infinite line power and NaN
#: audio, which is why :func:`check_sample` refuses it outright. A clipped draw
#: is re-checked against the sampler's guards and redrawn if the clip has made
#: it implausible, so the bank stays an honest draw from the bounded family
#: rather than a pile-up on the boundary.
AMP_EXP_BOUND = (4.398, 14.111)
FLOOR_EXP_BOUND = (0.0, 6.792)
EXPONENT_BOUND_SOURCE = "results/rig_sampler/structure.json:between_rig.per_fit (six fits)"


def clip_exponents(export: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Bound the speed exponents to the fitted range; name what was clipped.

    Applied to every export the bank builder draws around, and to the anchor
    itself wherever the anchor is rendered as a rig
    (``scripts/_model_matrix.py``'s point presets,
    ``notebooks/noise_lab.LegacyFit``), so the rendered rig is the one the arms
    were defined against — and so a whole-flight trajectory's ground phase does
    not meet a negative floor exponent.
    """
    out = dict(export)
    clipped: list[str] = []
    for key, (lo, hi) in (("amp_exp", AMP_EXP_BOUND), ("floor_exp", FLOOR_EXP_BOUND)):
        value = float(out.get(key, out["amp_exp"]))
        bounded = float(np.clip(value, lo, hi))
        if bounded != value:
            clipped.append(key)
        out[key] = bounded
    return out, clipped


#: The fields an entry takes from the FIT rather than from the donor policy's
#: per-clip draw. Everything not listed here (and not a ``fixed_*`` range in
#: :func:`entry_params`) is the stream's own draw, so the arms keep the base
#: policy's dynamics.
IDENTITY_FIELDS = (
    "profile_db",
    "harm_mean_db",
    "floor_ctrl_hz",
    "floor_ctrl_db",
    "floor_tilt_db_oct",
    "floor_mean_db",
    "floor_static_rel",
    "amp_rps_exponent",
    "amp_rps_exponent_floor",
    "amp_rps_ref",
    "coherence_k_half",
    "gamma_min_bins",
)


def donor_ranges(path: Path | str, index: int):
    """The per-clip draw ranges of one stochastic source of one policy.

    ``scripts/_build_rig_bank.py``'s ``--dynamics <policy.yaml>:<index>``:
    the DONOR of everything a bank entry does not take from its fit — the
    amplitude OU process, the floor level and tilt drift, the per-mic
    modulation, the shaft-jitter time constant and its per-clip spread, the
    phase diffusion and the label error.
    """
    import yaml

    from data_processing import stochastic_rotor_noise as srn

    policy = yaml.safe_load(Path(path).read_text())
    source = policy["sources"]["noise"][int(index)]
    if source.get("kind") != "stochastic":
        raise ValueError(f"{path}: source {index} is {source.get('kind')!r}, not stochastic")
    return srn.StochasticRanges.from_dict(source.get("ranges"))


def _as_tuple(value: np.ndarray | None) -> tuple[float, ...] | None:
    return None if value is None else tuple(np.asarray(value, dtype=np.float64).tolist())


def _as_nested(value: np.ndarray | None) -> tuple[tuple[float, ...], ...] | None:
    if value is None:
        return None
    return tuple(tuple(row) for row in np.asarray(value, dtype=np.float64).tolist())


def entry_params(
    export: dict[str, Any],
    ranges: Any,
    rng: np.random.Generator,
    *,
    rates: np.ndarray,
):
    """One bank entry: the stream's per-clip draw with the fitted rig on top.

    A stage-2 export describes ONE FITTED CLIP — its rig identity (timbre,
    floor shape, line widths, channel pattern, speed law) and nothing that
    varies inside a window, because :func:`stage2.params_from_export`
    deliberately zeroes every drift process. A training window needs both, so
    an entry is the donor policy's own per-clip draw
    (:func:`~data_processing.stochastic_rotor_noise.sample_params` on
    ``ranges``) with the fit's :data:`IDENTITY_FIELDS` written over it.

    The line WIDTH crosses over through the same seam the fitted policy uses:
    in ``line_mode: fm`` a line's width comes from the shaft's speed jitter and
    not from ``gamma``, and a Gaussian line with shaft-rate std ``sigma`` has
    HWHM ``sqrt(2 log 2) * k * sigma``, so the fit's ``gamma_slope`` is handed
    to the draw as ``fixed_shaft_jitter_rps = gamma_slope / GAUSS_HWHM``.
    Without it a bank would render a comb of dead-steady tones, which is not
    the fitted rig.

    ``rates`` sizes the order ladder (``params_from_export`` reads its minimum),
    so it is the SLOWEST speed the stream can ask for and not this clip's.
    """
    from dataclasses import replace

    from data_processing import stochastic_rotor_noise as srn
    from experiments.stochastic_fit import stage2

    fitted = stage2.params_from_export(
        export, rates, sample_rate=BANK_WORK_RATE, n_mics=BANK_N_MICS
    )
    # The widths and the channel pattern reach the draw through the very
    # `fixed_*` seams the base policy uses, so the per-clip width SPREAD
    # (`shaft_jitter_log_std`) still applies on top, exactly as it does there.
    ranges = replace(
        ranges,
        fixed_gamma0_hz=tuple(np.asarray(fitted.gamma0, dtype=np.float64).tolist()),
        fixed_gamma_slope_hz=tuple(np.asarray(fitted.gamma_slope, dtype=np.float64).tolist()),
        fixed_shaft_jitter_rps=tuple(
            (np.asarray(fitted.gamma_slope, dtype=np.float64) / GAUSS_HWHM).tolist()
        ),
        fixed_mic_gain_db=_as_nested(fitted.fixed_mic_gain_db),
        fixed_mic_floor_db=_as_tuple(fitted.fixed_mic_floor_db),
        fixed_mic_gain_all_db=_as_tuple(fitted.fixed_mic_gain_all_db),
    )
    draw = srn.sample_params(
        rng,
        ranges,
        n_rotors=fitted.n_rotors,
        n_harmonics=fitted.n_harmonics,
        sample_rate=BANK_WORK_RATE,
        line_bin_integrate=fitted.line_bin_integrate,
    )
    return draw.with_(**{name: getattr(fitted, name) for name in IDENTITY_FIELDS})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def print_structure_table(st: dict[str, Any]) -> None:
    """The compact phase A table."""
    p = st["provenance"]
    print("=" * 96)
    print("PHASE A - parameter neighbourhood of the pre-revision stage-2 rig fits")
    print("=" * 96)
    print(f"{'fit':24s} {'clips':>5s} {'R':>2s} {'K':>4s}  sha256[:12]  path")
    for name, v in p.items():
        print(
            f"{name:24s} {v['n_clips']:5d} {v['n_rotors']:2d} {v['n_orders']:4d}  "
            f"{v['sha256'][:12]}  {v['path']}"
        )

    c = st["decomposition"]["compare"]
    print(f"\n-- profile decomposition ({c['n_rotor_profiles']} rotor profiles) --")
    print(
        f"  residual std, one trend + one parity  [1,x,s]   : {c['resid_std_one_trend_plus_parity_mean']:5.2f} dB"
    )
    print(
        f"  residual std, separate even/odd trends         : {c['resid_std_even_odd_trends_mean']:5.2f} dB"
    )
    print(
        f"  residual std, decaying parity envelope (USED)  : {c['resid_std_envelope_mean']:5.2f} dB"
    )
    print(
        f"  envelope better than even/odd on {c['envelope_wins_over_even_odd']}/{c['n_rotor_profiles']} profiles"
    )

    r = st["residual_correlation"]
    print("\n-- residual correlation in log10 order --")
    print(
        f"  sq-exp length {r['length_sqexp_dec_median']:.4f} dec (err {r['err_sqexp_mean']:.4f})   "
        f"exp length {r['length_exp_dec_median']:.4f} dec (err {r['err_exp_mean']:.4f})"
    )
    print(
        f"  variogram length {r['length_variogram_dec_median']:.4f} dec (USED)   "
        f"ACF zero crossing {r['zero_crossing_dec_mean']:.3f} dec   "
        f"sq-exp wins {r['sqexp_better_than_exp']}/{st['decomposition']['compare']['n_rotor_profiles']}"
    )

    pa = st["parity"]
    print("\n-- parity (blade-passing) split --")
    print(
        f"  low-band amplitude  mean {pa['low_band_db_mean']:5.2f} dB  "
        f"range [{pa['low_band_db_min']:.2f}, {pa['low_band_db_max']:.2f}] dB"
    )
    print(f"  INVERTED low-band parity: {pa['n_inverted_low_band']} profiles (sign never flips)")

    t = st["trend"]
    print("\n-- trend drop from k=1 to k=K --")
    print(
        f"  min over all profiles {t['drop_db_min_all']:6.2f} dB   min over CRUISE only "
        f"{t['drop_db_min_cruise']:6.2f} dB   median {t['drop_db_median']:6.2f} dB"
    )
    print(f"  rising profiles: {t['n_rising']} {t['rising_profiles']}")

    bc = st["between_clip"]
    print("\n-- TIER 1, BETWEEN-CLIP (same rig, same regime, another window) --")
    print(f"  fields that move between clips: {bc['fields_varying_between_clips']}")
    print(
        f"  rotor gain      pooled rms {bc['rotor_gain_db']['pooled_rms']:6.3f} dB  "
        f"(median {bc['rotor_gain_db']['median']:.2f}, max {bc['rotor_gain_db']['max']:.2f}, "
        f"n={bc['rotor_gain_db']['n']})"
    )
    print(f"  floor_mean_db   pooled rms {bc['floor_mean_db']['pooled_rms']:6.3f} dB")
    print(f"  coherence (ln)  pooled rms {bc['coherence_k_half_ln']['pooled_rms']:6.3f}")
    print(
        f"  band level      pooled rms {bc['band_level_db']['pooled_rms']:6.3f} dB  "
        f"({LEVEL_BAND_HZ[0]:g}-{LEVEL_BAND_HZ[1]:g} Hz; the path sampler's own level "
        f"coordinate, vs the floor_mean_db FIELD above)"
    )
    print("  every other field: EXACTLY ZERO between-clip spread (tied by the rig fit)")

    br = st["between_refit"]
    print("\n-- TIER 2, BETWEEN-REFIT (same rig, same audio, another fit config) --")
    print(f"  {'quantity':22s} {'median|delta|':>14s} {'sigma':>9s}")
    for key, label in (
        ("slope_db_dec", "trend slope dB/dec"),
        ("parity_ln", "parity ln"),
        ("gamma_ln", "gamma ln"),
        ("floor_tilt_db_oct", "floor_tilt_db_oct"),
        ("floor_shape_db", "floor_shape_db"),
        ("mic_gain_db", "mic_gain_db"),
        ("mic_floor_db", "mic_floor_db"),
        ("gain_all_db", "gain_all_db"),
        ("amp_exp", "amp_exp"),
        ("floor_exp", "floor_exp"),
        ("floor_static_rel_ln", "floor_static_rel ln"),
    ):
        d = br[key]
        med = d.get("median_abs_delta", d.get("median_abs_dln", d.get("median_rms_delta")))
        print(f"  {label:22s} {med:14.4f} {d['sigma']:9.4f}")
    print(
        f"  {'resid redraw frac':22s} {br['resid_redraw_frac']['median']:14.4f} "
        f"{'(used directly)':>9s}"
    )
    print(
        f"  floor_shape lag-1 correlation across control points: "
        f"{br['floor_shape_db']['lag1_corr_mean']:.3f}"
    )

    bg = st["between_rig"]
    print("\n-- TIER 3, BETWEEN-RIG / BETWEEN-ROTOR (context, NOT the strength-1 width) --")
    print(
        f"  {'fit':24s} {'slope':>8s} {'parity':>8s} {'gamma0':>8s} {'g_slope':>8s} "
        f"{'amp_exp':>8s} {'fl_exp':>8s} {'coh':>6s}"
    )
    for name, v in bg["per_fit"].items():
        print(
            f"  {name:24s} {v['slope_db_dec_ols']:8.2f} {v['parity_low_db']:8.2f} "
            f"{v['gamma0']:8.3f} {v['gamma_slope']:8.3f} {v['amp_exp']:8.2f} "
            f"{v['floor_exp']:8.2f} {v['coherence_k_half']:6.2f}"
        )
    s = bg["std_of_fit_means"]
    print(
        f"  between-fit std   slope {s['slope_db_dec_ols']:.2f} dB/dec   "
        f"parity {s['parity_low_db']:.2f} dB   gamma_slope {s['gamma_slope']:.3f}"
    )
    w = bg["between_rotor_within_fit"]
    print(
        f"  between-rotor std slope {w['slope_db_dec_ols']:.2f} dB/dec   "
        f"parity ln {w['parity_ln']:.3f}   gain {w['gain_db']:.2f} dB"
    )

    lt = st["ltas_between_clip"]
    print("\n-- between-clip model LTAS deviation (context; NO LONGER the guard) --")
    print(
        f"  worst observed: {lt['max_rms_db']:.2f} dB rms, {lt['max_band_db']:.2f} dB "
        f"in the worst 1/3-octave band"
    )
    print(
        f"  between-refit worst observed: {lt['max_rms_db_refit']:.2f} dB rms, "
        f"{lt['max_band_db_refit']:.2f} dB band  (parameter differences COMPENSATE)"
    )
    print(
        f"  guard envelope is the REAL-PAIR difference instead: "
        f"{LTAS_ENVELOPE_X:g} x {REAL_PAIR_RMS_DB:g} dB rms in "
        f"{LEVEL_BAND_HZ[0]:g}-{LEVEL_BAND_HZ[1]:g} Hz, strength-independent, "
        f"override via ltas_tol"
    )


def print_width_table(st: dict[str, Any] | None = None) -> None:
    """The sampler's widths, frozen against freshly measured."""
    print("\n" + "=" * 96)
    print("PHASE B - sampler widths at strength 1 (frozen vs measured)")
    print("=" * 96)
    measured: dict[str, float] = {}
    if st is not None:
        br, bc, rc = st["between_refit"], st["between_clip"], st["residual_correlation"]
        measured = {
            "rotor_gain_db": bc["rotor_gain_db"]["pooled_rms"],
            "slope_db_dec": br["slope_db_dec"]["sigma"],
            "parity_ln": br["parity_ln"]["sigma"],
            "resid_redraw_frac": br["resid_redraw_frac"]["median"],
            "resid_length_dec": rc["length_variogram_dec_median"],
            "gamma_ln": br["gamma_ln"]["sigma"],
            "floor_mean_db": bc["floor_mean_db"]["pooled_rms"],
            "floor_tilt_db_oct": br["floor_tilt_db_oct"]["sigma"],
            "floor_shape_db": br["floor_shape_db"]["sigma"],
            "floor_shape_ar1": br["floor_shape_db"]["lag1_corr_mean"],
            "mic_gain_db": br["mic_gain_db"]["sigma"],
            "mic_floor_db": br["mic_floor_db"]["sigma"],
            "gain_all_db": br["gain_all_db"]["sigma"],
            "coherence_ln": bc["coherence_k_half_ln"]["pooled_rms"],
            "amp_exp": br["amp_exp"]["sigma"],
            "floor_exp": br["amp_exp"]["sigma"],
            "floor_static_rel_ln": br["floor_static_rel_ln"]["sigma"],
        }
    print(f"{'width':22s} {'frozen':>9s} {'measured':>9s} {'drift':>7s}  provenance")
    worst = 0.0
    for key in WIDTHS.__dataclass_fields__:
        frozen = float(getattr(WIDTHS, key))
        m = measured.get(key)
        if m is None:
            print(f"{key:22s} {frozen:9.4f} {'-':>9s} {'-':>7s}  {PROVENANCE[key]}")
            continue
        drift = abs(frozen - m) / max(abs(m), 1e-12)
        worst = max(worst, drift)
        print(f"{key:22s} {frozen:9.4f} {m:9.4f} {100 * drift:6.2f}%  {PROVENANCE[key]}")
    print(
        f"\nguards: amp_exp >= 0 and floor_exp >= 0 | trend drop >= {TREND_MARGIN_DB:g} dB | "
        f"gamma excursion <= {GAMMA_EXCURSION_CAP:g}x | LTAS rms <= "
        f"{LTAS_ENVELOPE_X:g} x {REAL_PAIR_RMS_DB:g} dB (real-pair, "
        f"{LEVEL_BAND_HZ[0]:g}-{LEVEL_BAND_HZ[1]:g} Hz), strength-independent"
    )
    if measured:
        print(f"worst frozen-vs-measured drift: {100 * worst:.2f}%")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--root", default=".", help="repo root")
    ap.add_argument("--out", default=str(STRUCTURE_PATH))
    args = ap.parse_args(argv)

    st = measure_structure(args.root)
    out = Path(args.root) / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(st, indent=1))
    print_structure_table(st)
    print_width_table(st)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
