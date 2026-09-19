"""The exact discrete-time renderer of the v2 model, in ABSOLUTE units.

``render_noise(fit, rps_rev_s, sr=16000, n_mics=8, seed=0) -> (n_mics, T)``.

The last campaign left the absolute level open because its renderer defaulted
to RMS normalisation and only the evaluator's ``normalize_rms=None`` path
avoided it (``revised-phase-campaign.qmd:37-42,262-277``). There is no such
wrapper here: the renderer HAS no normalisation, the level is the fitted one,
and ``test_render_level_matches_model`` reads the rendered periodogram back with
:func:`revised_eval.window_periodogram` and compares it to the model ``M`` the
fit reports. If those two ever disagree, the renderer is wrong — not the
convention.

WHAT IS DRAWN, per rotor ``r`` and order ``k`` (the total phase of the model,
``docs/explainers/noise-model-v2-plan.qmd``, "Total phase"):

    phi_mrk(t) = 2 pi k int_0^t f_r + k theta_r(t) + eps_rk(t) + alpha_mrk

* ``(theta_r, nu_r)`` — ONE fresh integrated-OU shaft draw per rotor at the
  work rate, through :func:`revised_phase.simulate_state`'s EXACT transition;
* ``psi_rk`` — an INDEPENDENT WIENER phase per line, drawn as the random walk
  of Model R3: increments ``N(0, 4 pi gamma_rk dt)``, cumulatively summed, so
  ``Var[psi(t + tau) - psi(t)] = 4 pi gamma_rk |tau|`` and the line is a
  Lorentzian of half-width ``gamma_rk`` Hz — exactly the factor
  ``exp(-2 pi gamma_rk |tau|)`` :func:`.lag.r_tau` assumes, which
  ``test_render_reproduces_the_per_order_lag_law`` measures by demodulating
  one order out of the render;
* ``alpha_mrk`` — the fixed per-(rotor, order) microphone response phase, drawn
  once per render and CONSTANT in time, exactly as
  :func:`revised_phase.render_revised` draws it and for the same reason (it is
  not identifiable from an expected periodogram, and the model's independent
  uniform initial phase is what makes the components add in power).

ORDERS. Only orders whose line sits below the OUTPUT Nyquist are rendered, the
same cap the forward model uses (:func:`.spectrum.k_max_for_carrier`), so there
is nothing to alias; the render still goes through
:func:`stage2.antialias` and the unchanged :func:`clips.decimate`, because that
pair IS the chain :func:`stage2.render_transfer_power` describes to the fit and
the fitted ``M`` carries it.

THE FLOOR is white noise shaped by the square root of the SHARED
:func:`revised_phase.floor_power_spectrum`, with the floor's own speed envelope
— the same spectrum the fit reads and the same envelope it evaluates, so the
fitted floor and the synthesised floor cannot drift apart.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, overload

import numpy as np

from experiments.stochastic_fit.data import Clip
from experiments.stochastic_fit.revised_phase import (
    AMP_RPS_REF,
    SPEED_FLOOR_RPS,
    floor_geometry,
    floor_power_spectrum,
    simulate_state,
)
from experiments.stochastic_fit.stage2 import antialias

from . import READABLE_FIT_SCHEMAS
from . import model as MD
from . import spectrum as SP

__all__ = [
    "expected_periodogram",
    "expected_periodogram_regimes",
    "regime_blend_weight",
    "regime_seeds",
    "render_noise",
    "render_noise_regimes",
]


#: The fit schemas the renderer reads: its own, and R1/R2's, whose per-order
#: OU :func:`.model.gamma_from_params` maps onto an equivalent width.
READABLE_SCHEMAS = READABLE_FIT_SCHEMAS


def _check_schema(fit: dict[str, Any]) -> dict[str, Any]:
    schema = fit.get("schema")
    if schema not in READABLE_SCHEMAS:
        raise ValueError(f"expected schema in {READABLE_SCHEMAS}, got {schema!r}")
    return fit["params"]


@overload
def render_noise(
    fit: dict[str, Any],
    rps_rev_s: np.ndarray,
    *,
    sr: int = SP.FLIGHT_SR,
    n_mics: int = 8,
    seed: int = 0,
    sr_work: int = SP.SAMPLE_RATE_WORK,
    return_diagnostics: Literal[False] = False,
) -> np.ndarray: ...


@overload
def render_noise(
    fit: dict[str, Any],
    rps_rev_s: np.ndarray,
    *,
    sr: int = SP.FLIGHT_SR,
    n_mics: int = 8,
    seed: int = 0,
    sr_work: int = SP.SAMPLE_RATE_WORK,
    return_diagnostics: Literal[True],
) -> tuple[np.ndarray, dict[str, Any]]: ...


def render_noise(
    fit: dict[str, Any],
    rps_rev_s: np.ndarray,
    *,
    sr: int = SP.FLIGHT_SR,
    n_mics: int = 8,
    seed: int = 0,
    sr_work: int = SP.SAMPLE_RATE_WORK,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Synthesise the fitted noise on the carriers ``rps_rev_s``.

    Parameters
    ----------
    fit
        A ``noise-v2-fit/1`` payload (:func:`.fit.write_fit`'s output).
    rps_rev_s
        ``(R, T)`` per-rotor carrier in rev/s on the OUTPUT grid at ``sr``. A
        bench fit's own constant carriers are NOT used here: the renderer is
        driven by the carriers it is asked for, which is what lets a bench-fitted
        rig be rendered on a flight trajectory.
    sr, n_mics, seed
        Output rate, channel count and the single seed of every draw.

    Returns
    -------
    ``(n_mics, T)`` float64 in ABSOLUTE units — the same units the fitted ``M``
    and :func:`revised_eval.window_periodogram` are in. No RMS normalisation.
    """
    p = _check_schema(fit)
    rps = np.atleast_2d(np.asarray(rps_rev_s, dtype=np.float64))
    n_rotors, n_out = rps.shape
    profile_db = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
    if profile_db.shape[0] != n_rotors:
        if profile_db.shape[0] == 1:
            profile_db = np.repeat(profile_db, n_rotors, axis=0)
        else:
            raise ValueError(
                f"fit carries {profile_db.shape[0]} rotor profiles, asked to render {n_rotors}"
            )
    # (R, K) widths: a /2 payload's own, or a /1 payload's per-order OU mapped
    # onto the equivalent Lorentzian (model.gamma_from_params)
    gamma_hz = MD.gamma_from_params(p)
    if gamma_hz.shape[0] != n_rotors:
        if gamma_hz.shape[0] == 1:
            gamma_hz = np.repeat(gamma_hz, n_rotors, axis=0)
        else:
            raise ValueError(
                f"fit carries {gamma_hz.shape[0]} rotor widths, asked to render {n_rotors}"
            )

    oversample = int(sr_work) // int(sr)
    if oversample * int(sr) != int(sr_work):
        raise ValueError(f"sr_work {sr_work} must be an integer multiple of sr {sr}")
    n_work = n_out * oversample
    dt = 1.0 / float(sr_work)
    t_src = np.arange(n_out) / float(sr)
    t_work = np.arange(n_work) / float(sr_work)
    f0 = np.maximum(
        np.stack([np.interp(t_work, t_src, r) for r in rps]), SPEED_FLOOR_RPS
    )  # (R, n_work)

    k_max = min(
        int(profile_db.shape[1]),
        SP.k_max_for_carrier(f0.max(axis=1), sr, k_cap=int(profile_db.shape[1])),
    )
    if k_max < 1:
        raise ValueError(f"no order of a {float(f0.max()):.1f} rev/s rotor fits below {sr / 2} Hz")

    ss = np.random.SeedSequence(int(seed))
    rng_state, rng_psi, rng_alpha, rng_floor = (np.random.default_rng(s) for s in ss.spawn(4))

    lam = float(p["lam"])
    sigma_nu = float(p["sigma_nu"])
    innov = rng_state.standard_normal((n_rotors, n_work, 2))
    theta, _nu = simulate_state(innov, lam=lam, sigma=sigma_nu, dt=dt)
    phase = 2.0 * np.pi * np.cumsum(f0, axis=1) * dt + theta  # (R, n_work)

    mic_line_db = np.asarray(p["profile"]["mic_line_gain_db"], dtype=np.float64)
    mic_floor_db = np.asarray(p["floor"]["mic_floor_db"], dtype=np.float64)
    gain_all_db = np.asarray(p["mic_gains_db"], dtype=np.float64)
    for name, arr in (
        ("mic_line_gain_db", mic_line_db),
        ("mic_floor_db", mic_floor_db),
        ("mic_gains_db", gain_all_db),
    ):
        if arr.shape[0] < n_mics:
            raise ValueError(
                f"fit carries {arr.shape[0]} microphones in {name}, asked for {n_mics}"
            )
    mic_line_db = mic_line_db[:n_mics, :n_rotors] if mic_line_db.ndim == 2 else mic_line_db[:n_mics]
    line_gain = 10.0 ** ((mic_line_db - mic_line_db.mean(axis=0, keepdims=True)) / 10.0)
    all_gain = 10.0 ** ((gain_all_db[:n_mics] - gain_all_db[:n_mics].mean()) / 10.0)

    amp_exp = float(p["profile"]["amp_exp"])
    speed = f0 / AMP_RPS_REF
    audio = np.zeros((n_mics, n_work), dtype=np.float64)
    for r in range(n_rotors):
        line_amp = np.sqrt(2.0 * 10.0 ** (profile_db[r, :k_max] / 10.0))
        speed_amp = np.sqrt(speed[r] ** amp_exp)
        for k in range(1, k_max + 1):
            # the line's own Wiener phase: increments N(0, 4 pi gamma dt),
            # cumulatively summed. Its increment variance IS the exponent the
            # lag law carries, so the rendered line is the Lorentzian of
            # half-width gamma_rk the fit was written in terms of.
            step = math.sqrt(4.0 * math.pi * max(float(gamma_hz[r, k - 1]), 0.0) * dt)
            psi = np.cumsum(rng_psi.standard_normal(n_work) * step)
            arg = k * phase[r] + psi
            env = line_amp[k - 1] * speed_amp
            alpha = rng_alpha.uniform(0.0, 2.0 * np.pi, size=n_mics)
            # cos(arg + alpha_m) by angle addition: TWO full-length trig passes
            # per (rotor, order) instead of n_mics of them.
            ec = env * np.cos(arg)
            es = env * np.sin(arg)
            for m in range(n_mics):
                g = math.sqrt(line_gain[m, r]) if line_gain.ndim == 2 else math.sqrt(line_gain[m])
                audio[m] += (g * math.cos(alpha[m])) * ec
                audio[m] -= (g * math.sin(alpha[m])) * es

    shape_mat, tilt_oct = floor_geometry(np.fft.rfftfreq(n_work, d=dt), SP.floor_ctrl_hz(sr))
    shape_db = SP.floor_shape_db(p["floor"]["floor_shape_z"], sr=sr)
    floor_psd = floor_power_spectrum(
        shape_mat,
        tilt_oct,
        mean_db=float(p["floor"]["floor_mean_db"]),
        ctrl_db=shape_db,
        tilt_db_oct=float(p["floor"]["floor_tilt_db_oct"]),
        rate_factor=float(sr_work) / float(sr),
    )
    floor_gain_t = (speed ** float(p["floor"]["floor_exp"])).mean(axis=0) + float(
        p["floor"]["floor_static_rel"]
    )
    floor_amp = np.sqrt(floor_psd)
    for m in range(n_mics):
        white = rng_floor.standard_normal(n_work)
        shaped = np.fft.irfft(np.fft.rfft(white) * floor_amp, n=n_work)
        audio[m] += shaped * np.sqrt(floor_gain_t) * math.sqrt(10.0 ** (mic_floor_db[m] / 10.0))
    audio *= np.sqrt(all_gain)[:, None]

    from experiments.stochastic_fit import clips as C

    out = C.decimate(
        Clip("noise_v2_render", "synthetic", antialias(audio, sr_work), f0, int(sr_work)),
        int(sr),
    )
    rendered = np.asarray(out.audio, dtype=np.float64)[:, :n_out]
    if rendered.shape[1] < n_out:
        raise ValueError(f"render produced {rendered.shape[1]} samples, asked for {n_out}")
    if not return_diagnostics:
        return rendered
    return rendered, dict(
        seed=int(seed),
        sr=int(sr),
        sr_work=int(sr_work),
        n_orders=int(k_max),
        n_rotors=int(n_rotors),
        n_mics=int(n_mics),
        rms=np.sqrt((rendered**2).mean(axis=1)).tolist(),
        normalisation="none (absolute fitted level)",
    )


def expected_periodogram(
    fit: dict[str, Any],
    rps_rev_s: np.ndarray,
    *,
    n_fft: int = SP.FLIGHT_N_FFT,
    hop: int = SP.FLIGHT_HOP,
    sr: int = SP.FLIGHT_SR,
    n_mics: int = 8,
    frame_chunk: int = 16,
    sr_work: int = SP.SAMPLE_RATE_WORK,
) -> np.ndarray:
    """``(n_mics, n_frames, n_fft // 2 + 1)`` predicted ``M`` on a moving carrier.

    The v2 analogue of :func:`revised_phase.predict_spectrum`: the fitted comb
    and floor evaluated through the FLIGHT forward model on the real carrier,
    in ``data.periodogram`` units, with the frame grid
    ``arange(1 + (T - n_fft) // hop) * hop`` — ``data.periodogram``'s own
    framing, so bin ``j`` is ``j sr / n_fft`` and frame centre ``i`` is
    ``(start_i + n_fft / 2) / sr``, exactly the support's.
    """
    import torch

    p = _check_schema(fit)
    rps = np.atleast_2d(np.asarray(rps_rev_s, dtype=np.float64))
    n = int(rps.shape[1])
    if n < n_fft:
        raise ValueError(f"{n} samples is shorter than n_fft {n_fft}")
    starts = np.arange(1 + (n - int(n_fft)) // int(hop)) * int(hop)
    grid = SP.flight_grid(sr=sr, n_fft=n_fft, hop=hop, sr_work=sr_work)
    params = MD.params_from_dict(p)
    prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
    k_max = min(
        int(prof.shape[1]),
        SP.k_max_for_carrier(rps.max(axis=1), sr, k_cap=int(prof.shape[1])),
    )
    out = np.empty((int(n_mics), starts.size, int(n_fft) // 2 + 1), dtype=np.float64)
    with torch.no_grad():
        for c0 in range(0, starts.size, int(frame_chunk)):
            sl = slice(c0, min(c0 + int(frame_chunk), starts.size))
            rate = SP.flight_rate_work(grid, rps, starts[sl])
            m = SP.flight_model(grid, params, rate_work=rate, k_max=k_max)
            got = m.cpu().numpy()
            if got.shape[0] < n_mics:
                raise ValueError(f"fit carries {got.shape[0]} microphones, asked for {n_mics}")
            out[:, sl, :] = got[:n_mics]
    return out


# ── the per-regime composition ──────────────────────────────────────────────

#: The regimes :func:`render_noise_regimes` composes, SLOWEST FIRST. There is
#: no ``ramp`` entry: the ramp is the interpolation between these two, not a
#: third fit (the previous generation had no ramp export either).
REGIME_ORDER: tuple[str, ...] = ("standby", "cruise")


def regime_blend_weight(
    rps_rev_s: np.ndarray, *, sr: int = SP.FLIGHT_SR, policy: Any = None
) -> np.ndarray:
    """``(T,)`` weight on the CRUISE fit, in [0, 1] — the gating rule itself.

    This is :func:`data_processing.rps_gating.regime_weight`, the previous
    generation's own regime rule, with its ``settle_s`` term switched off: the
    smoothstep ``3x^2 - 2x^3`` in the SLOWEST rotor's rate, exactly 0 at
    ``STANDBY_MAX_RPS`` = 45 rev/s and exactly 1 at ``CRUISE_MIN_RPS`` = 65
    rev/s, so the two thresholds here and in every refined label are the same
    constants. ``settle_s`` exists to delay TRUST in a refined label for a
    second after a spool-up; a model's parameters have no such history, and
    keeping it would make this weight depend on how much of the recording came
    before the window rather than on the carrier alone.
    """
    from data_processing.rps_gating import GatePolicy, regime_weight

    rps = np.atleast_2d(np.asarray(rps_rev_s, dtype=np.float64))
    ft = np.arange(rps.shape[1], dtype=np.float64) / float(sr)
    return np.asarray(
        regime_weight(ft, rps, policy if policy is not None else GatePolicy(settle_s=0.0)),
        dtype=np.float64,
    )


def regime_seeds(seed: int, regimes: Sequence[str] = REGIME_ORDER) -> dict[str, int]:
    """One INDEPENDENT stream per regime, spawned from the arm's single seed.

    The regimes' renders are summed inside the blend, so they must not share a
    stream: with the same seed the two renders' shaft and line phases are
    correlated and the sum would gain up to 3 dB where both weights are near
    1/2, which is exactly the level artefact the blend exists to avoid. Spawned
    from one ``SeedSequence`` they are independent, so their POWERS add and the
    blend interpolates the two regimes' levels (see
    :func:`render_noise_regimes`). The mapping is deterministic in ``seed``,
    which is what lets a single-regime segment of a composed render be
    reproduced by :func:`render_noise` alone.
    """
    children = np.random.SeedSequence(int(seed)).spawn(len(regimes))
    return {
        str(name): int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(regimes, children, strict=True)
    }


def _regime_fits(fits_by_regime: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    missing = [r for r in REGIME_ORDER if r not in fits_by_regime]
    if missing:
        raise ValueError(f"the regime composition needs a fit per regime; missing {missing}")
    extra = [r for r in fits_by_regime if r not in REGIME_ORDER]
    if extra:
        raise ValueError(f"unknown regime(s) {extra}; this composition is over {REGIME_ORDER}")
    return {r: fits_by_regime[r] for r in REGIME_ORDER}


def render_noise_regimes(
    fits_by_regime: Mapping[str, dict[str, Any]],
    rps_rev_s: np.ndarray,
    *,
    sr: int = SP.FLIGHT_SR,
    n_mics: int = 8,
    seed: int = 0,
    sr_work: int = SP.SAMPLE_RATE_WORK,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Synthesise ONE carrier track from a PER-REGIME pair of fits.

    ``fits_by_regime`` carries one payload per entry of :data:`REGIME_ORDER` —
    a standby-only fit and a cruise-only fit of the same rig. Each is rendered
    on the WHOLE track by :func:`render_noise` (so each regime's own render is
    reproduced sample for sample where it is the only contributor, with
    ``seed=regime_seeds(seed)[regime]``), and the two are combined per sample:

        out = sqrt(1 - w) * standby + sqrt(w) * cruise,
        w = regime_blend_weight(rps) -> 0 at <= 45 rev/s, 1 at >= 65 rev/s

    so that, the two streams being independent, the composed POWER is exactly
    ``(1 - w) P_standby + w P_cruise``: the standby fit alone below
    ``rps_gating.STANDBY_MAX_RPS``, the cruise fit alone at or above
    ``rps_gating.CRUISE_MIN_RPS``, and a continuous interpolation of the two
    regimes' LEVELS across the ramp band in between. The weight is a
    smoothstep in the SLOWEST rotor's rate, so the composition is continuous
    and has zero slope at both thresholds; nothing is filtered in time.

    WHY THIS RULE. The previous-generation arm had no pooled fit either: it
    selected a per-regime stage-2 export and the regime bands are the same 45 /
    65 rev/s constants (``data_processing.rps_gating``, whose smoothstep is
    reused here verbatim). Its selection was a HARD switch on the support's
    DECLARED regime label — standby took ``results/S2/standby.json`` and BOTH
    ramp and cruise took the cruise export
    (``noise_v2_round_score.LEGACY_BASELINE``, ``from_regime: cruise``) — which
    needs a label per window and cannot render a track that crosses a band at
    all. This composition keeps the legacy per-regime PARAMETERS and the legacy
    thresholds and replaces the label lookup by the carrier itself, so the ramp
    is interpolated between the two fitted regimes instead of being handed to
    cruise whole.

    A regime whose weight is zero everywhere is NOT rendered: a cruise-only
    window costs exactly one render, and only a window that actually crosses
    the band costs two.
    """
    fits = _regime_fits(fits_by_regime)
    rps = np.atleast_2d(np.asarray(rps_rev_s, dtype=np.float64))
    w = regime_blend_weight(rps, sr=sr)
    gains = {"standby": np.sqrt(np.maximum(1.0 - w, 0.0)), "cruise": np.sqrt(np.maximum(w, 0.0))}
    seeds = regime_seeds(int(seed), REGIME_ORDER)
    out = np.zeros((int(n_mics), rps.shape[1]), dtype=np.float64)
    per_regime: dict[str, Any] = {}
    for regime in REGIME_ORDER:
        gain = gains[regime]
        if not bool(np.any(gain > 0.0)):
            per_regime[regime] = dict(rendered=False, seed=seeds[regime], weight_max=0.0)
            continue
        audio = np.asarray(
            render_noise(
                fits[regime],
                rps,
                sr=sr,
                n_mics=n_mics,
                seed=seeds[regime],
                sr_work=sr_work,
            ),
            dtype=np.float64,
        )
        out += audio * gain[None, :]
        per_regime[regime] = dict(
            rendered=True,
            seed=seeds[regime],
            weight_max=float(gain.max() ** 2),
            rms=np.sqrt((audio**2).mean(axis=1)).tolist(),
        )
    if not return_diagnostics:
        return out
    return out, dict(
        seed=int(seed),
        sr=int(sr),
        sr_work=int(sr_work),
        n_mics=int(n_mics),
        n_rotors=int(rps.shape[0]),
        regimes=list(REGIME_ORDER),
        per_regime=per_regime,
        cruise_weight=dict(
            min=float(w.min()),
            max=float(w.max()),
            mean=float(w.mean()),
            fraction_in_blend=float(np.mean((w > 0.0) & (w < 1.0))),
        ),
        rms=np.sqrt((out**2).mean(axis=1)).tolist(),
        rule=(
            "per-sample sqrt-weight sum of the standby and cruise renders; the weight is "
            "rps_gating.regime_weight (smoothstep in the slowest rotor, 0 at 45 rev/s, 1 at 65) "
            "with settle_s disabled, so the composed POWER interpolates the two regimes' levels"
        ),
        normalisation="none (absolute fitted level)",
    )


def expected_periodogram_regimes(
    fits_by_regime: Mapping[str, dict[str, Any]],
    rps_rev_s: np.ndarray,
    *,
    n_fft: int = SP.FLIGHT_N_FFT,
    hop: int = SP.FLIGHT_HOP,
    sr: int = SP.FLIGHT_SR,
    n_mics: int = 8,
    frame_chunk: int = 16,
    sr_work: int = SP.SAMPLE_RATE_WORK,
) -> np.ndarray:
    """The composition's expected ``M``: the same power blend, per FRAME.

    ``(1 - w_i) M_standby + w_i M_cruise`` with ``w_i`` the composed weight at
    frame ``i``'s CENTRE sample, which is the expectation of what
    :func:`render_noise_regimes` draws (the two streams are independent, so
    their powers add) and therefore the predicted periodogram the likelihood
    gate must read for this candidate.
    """
    fits = _regime_fits(fits_by_regime)
    rps = np.atleast_2d(np.asarray(rps_rev_s, dtype=np.float64))
    w = regime_blend_weight(rps, sr=sr)
    starts = np.arange(1 + (int(rps.shape[1]) - int(n_fft)) // int(hop)) * int(hop)
    centres = np.clip(starts + int(n_fft) // 2, 0, w.size - 1)
    w_frame = w[centres]
    out: np.ndarray | None = None
    for regime, weight in (("standby", 1.0 - w_frame), ("cruise", w_frame)):
        if not bool(np.any(weight > 0.0)):
            continue
        m = expected_periodogram(
            fits[regime],
            rps,
            n_fft=n_fft,
            hop=hop,
            sr=sr,
            n_mics=n_mics,
            frame_chunk=frame_chunk,
            sr_work=sr_work,
        )
        term = np.asarray(m, dtype=np.float64) * weight[None, :, None]
        out = term if out is None else out + term
    if out is None:  # pragma: no cover - the weights sum to one by construction
        raise ValueError("no regime carries any weight on this carrier track")
    return out
