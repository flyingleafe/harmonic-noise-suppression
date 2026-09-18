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

__all__ = ["expected_periodogram", "render_noise"]


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
