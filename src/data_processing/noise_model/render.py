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

ORDERS. Every order of the profile whose line sits below the WORK Nyquist is
rendered (:func:`.spectrum.k_max_for_carrier` at ``sr_work``); the ones above
the OUTPUT Nyquist, or crossing it as the carrier moves, are removed by
:func:`.resample.antialias` and the unchanged :func:`.resample.decimate_audio`,
because that pair IS the chain
:func:`experiments.stochastic_fit.stage2.render_transfer_power` describes to the
fit and the fitted ``M`` carries it. A real comb rolls off through the
recorder's transition band with its last orders still well over the floor; a
cap at the output Nyquist for the clip's fastest carrier would cut it with a
cliff, so the work rate must stay above the output rate.

THE FLOOR is white noise shaped by the square root of the SHARED
:func:`.floor.floor_power_spectrum`, with the floor's own speed envelope
— the same spectrum the fit reads and the same envelope it evaluates, so the
fitted floor and the synthesised floor cannot drift apart.

A ``noise-v3-fit/1`` PAYLOAD (``docs/explainers/noise-model-v3-wander.qmd``)
renders through the same path with three differences, all read off the
payload: no microphone block (the channels were normalised in the data, so
every mic has unit gain), the floor's control values scaled by the measured
``sigma_B`` with no tilt, and — the point of v3 — the slow amplitude WANDER.
The per-window latents a fit estimated are nuisance and are NOT rendered;
the rig's measured ``(sigma, tau, block_s)`` are, as FRESH stationary OU
tracks drawn per clip at the block centres (:func:`.v3.ou_blocks`) and
interpolated linearly in dB between them: ``d_r + v_rk`` multiplies line
``(r, k)``'s power by ``10^{(d + v)/10}``, and ``u + sum_j B_j(f) u_j``
multiplies the floor's through a WOLA short-time gain (:func:`_slow_gain`).
DREGON's static per-mic wind term ``W_m(f)`` (:func:`.v3.wind_shape`) is its
own independent shaped noise, added unscaled by speed. The v3 draws come
from streams spawned AFTER the four v2 ones, so a v2 payload renders exactly
as before.

WANDER MEAN. Every wander track is drawn ZERO-MEAN in dB by default
(``wander_mean="zero"``), so a family of sd ``s`` raises the mean power it
multiplies by ``exp((s ln10 / 10)^2 / 2)``. ``wander_mean="power"`` draws each
of the four with mean ``-s^2 ln10 / 20`` dB instead — ``d`` and ``u`` at
their scalar sd, every line's ``v`` at its own order group's sd, every
colour point ``u_j`` at ``sigma_uj sqrt((A A^T)_jj)`` when the colour is the
kernel mix ``A y`` — so ``E[10^{x/10}] = 1`` at every block centre (and
colour control point) and the rendered mean power is the fitted one there.
The same draws are used, only shifted. Between block centres the linear
interpolation in dB has less variance than ``s^2`` — short by
``(1 - rho) s^2 / 3`` averaged over a block, ``rho`` the block-to-block
correlation — so there the mean power sits slightly BELOW the fitted one.

A ``noise-v4-fit/3`` PAYLOAD (``docs/explainers/noise-model-v4.qmd``) is drawn
by :func:`_render_noise_v4` from the laws :mod:`.v4` reads: per line a
bounded log-amplitude OU and an independent bounded PHASE OU (no Wiener
phase), motor lines at ``k = 3 p n`` on the shaft alone, the line powers on
``s^alpha``, the floor as the sum of per-rotor floors ``s_r^{b(f)}`` through
the WOLA gain, and the rotor-to-mic response ``E_rm(f)``. It shares the line
kernel, the anti-alias chain and the WOLA with v2/v3, which render
bit-identically to before.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, overload

import numpy as np
from numba import njit

from data_processing.noise_model import FIT_SCHEMA_V3, FIT_SCHEMA_V4, READABLE_FIT_SCHEMAS
from data_processing.noise_model import spectrum as SP
from data_processing.noise_model.constants import AMP_RPS_REF, SPEED_FLOOR_RPS
from data_processing.noise_model.floor import floor_geometry, floor_power_spectrum
from data_processing.noise_model.ou import simulate_state
from data_processing.noise_model.params import array_response_db, gamma_from_params
from data_processing.noise_model.params import check_schema as _check_schema
from data_processing.noise_model.resample import AA_PASS_HZ, antialias, decimate_audio
from data_processing.noise_model.v3 import Wander, ou_blocks, wind_shape
from data_processing.noise_model.v4 import WORK_RATE_HZ as V4_WORK_RATE_HZ
from data_processing.noise_model.v4 import V4Params

__all__ = [
    "REGIME_ORDER",
    "READABLE_SCHEMAS",
    "fit_work_rate",
    "regime_blend_weight",
    "regime_seeds",
    "render_noise",
    "render_noise_regimes",
]


#: The fit schemas the renderer reads: its own, R1/R2's, whose per-order OU
#: :func:`.params.gamma_from_params` maps onto an equivalent width, v3's, and
#: v4's (:mod:`.v4`).
READABLE_SCHEMAS = READABLE_FIT_SCHEMAS + (FIT_SCHEMA_V4,)

#: WOLA frame (work samples) of the floor wander's short-time gain: 64 ms at
#: 64 kHz, hop half of it. The gain it applies moves on the block scale
#: (hundreds of ms) and is smooth in log-frequency, so 15.6 Hz bins and 32 ms
#: steps resolve it; a sqrt-Hann pair at 50 % overlap reconstructs a constant
#: gain exactly.
WANDER_WOLA_FRAME = 4096

#: WOLA frame of v4's floor speed law ``sum_r 10^{E_rm(f)/10} s_r(t)^{b(f)}``,
#: at the OUTPUT rate the floor is synthesised at: 64 ms at 16 kHz, hop 32 ms,
#: 15.6 Hz bins. The gain follows the carriers (seconds) and is smooth in
#: frequency (``b(f)`` is linear in log2 f, the pair EQ is a 200 Hz smooth).
V4_FLOOR_WOLA_FRAME = 1024


#: ``render_noise``'s ``wander_mean``: the wander tracks' mean in dB, ``"zero"``
#: (the fitted model's latents at their prior mean) or ``"power"`` (each shifted
#: by ``-sigma^2 ln10 / 20`` so its power multiplier has mean one).
WanderMean = Literal["zero", "power"]
WANDER_MEANS: tuple[str, ...] = ("zero", "power")


#: The LINE ENVELOPE grid: every slow per-line factor (speed law, wander, the
#: AM pedestal, the Wiener phase, v4's amplitude and phase OUs) is drawn or sampled at this rate and
#: interpolated linearly onto the work grid by :func:`_line_kernel`. 1 kHz
#: resolves a 25 Hz pedestal and a Lorentzian of tens of Hz; the carriers
#: themselves (``k phi_r``) stay at the work rate.
ENV_RATE_HZ = 1000
#: Work samples per :func:`_line_kernel` block (``2 K x block`` float32 live).
LINE_BLOCK = 4096


def _ou_paths(
    rng: np.random.Generator | None,
    sigma2: np.ndarray,
    gamma_hz: np.ndarray,
    n_env: int,
    dt_env: float,
    dtype: Any = np.float64,
) -> np.ndarray:
    """``(R, K, n_env)`` STATIONARY OU paths on the envelope grid, one per
    (rotor, order): variance ``sigma2``, autocorrelation ``exp(-2 pi gamma_hz |tau|)``."""
    assert rng is not None
    R, K = sigma2.shape
    rho = np.exp(-2.0 * np.pi * gamma_hz * dt_env).astype(dtype, copy=False)  # (R, K)
    sd = np.sqrt(np.maximum(sigma2, 0.0)).astype(dtype, copy=False)
    g = np.empty((R, K, n_env), dtype=dtype)
    g[:, :, 0] = sd * rng.standard_normal((R, K), dtype=dtype)
    drive = (
        sd[:, :, None]
        * np.sqrt(1.0 - rho[:, :, None] ** 2)
        * rng.standard_normal((R, K, n_env), dtype=dtype)
    )
    _ar1(g, rho, drive)
    return g


def _am_envelopes(
    rng: np.random.Generator | None,
    sigma2: np.ndarray,
    gamma_hz: np.ndarray,
    n_env: int,
    dt_env: float,
) -> np.ndarray:
    """``(R, K, n_env)`` log-OU amplitude envelopes ``exp(g - sigma2/2)`` (unit
    MEAN AMPLITUDE: the profile stays the coherent core, the pedestal adds
    ``e^{sigma2} - 1`` of it) on the envelope grid, one per (rotor, order),
    shared by the mics: ``g`` has variance ``sigma2`` and rate ``2 pi gamma_hz``."""
    g = _ou_paths(rng, sigma2, gamma_hz, n_env, dt_env)
    return np.exp(g - 0.5 * sigma2[:, :, None])


@njit(cache=True, fastmath=True)
def _line_kernel(
    c1: np.ndarray,
    s1: np.ndarray,
    c0: np.ndarray,
    s0: np.ndarray,
    k_init: int,
    e_re: np.ndarray,
    e_im: np.ndarray,
    hop_env: int,
    t0: int,
    n: int,
    zbuf: np.ndarray,
    ck: np.ndarray,
    sk: np.ndarray,
    orders: np.ndarray,
) -> None:
    """Fill ``zbuf[:, :n]`` (``(2 L, block)`` float32) with ``Re`` and ``Im`` of
    ``Z_l(t) = E_l(t) e^{i k_l phi(t)}`` for work samples ``t0 .. t0 + n`` and
    the ``L`` lines of orders ``orders`` (ascending, ``>= k_init``; an order
    may repeat, e.g. v4's aerodynamic and motor lines at ``k = 3 p n``):
    ``cos / sin(k phi)`` by the angle-addition recurrence in ``k`` from
    ``(c0, s0) = (cos, sin)(k_init phi)`` in steps of ``(c1, s1) = (cos phi,
    sin phi)`` (``ck``, ``sk``: block-length scratch), ``E_l`` linearly
    interpolated from the envelope grid (``hop_env`` work samples per knot;
    ``t0`` is a knot). Loops are line outer, ``t`` inner within a knot
    segment: every ``t`` is an independent lane."""
    L = e_re.shape[0]
    inv = np.float32(1.0 / hop_env)
    for i in range(n):
        ck[i] = c0[t0 + i]
        sk[i] = s0[t0 + i]
    k = k_init
    j0 = t0 // hop_env
    n_seg = (n + hop_env - 1) // hop_env
    for line in range(L):
        while k < orders[line]:
            for i in range(n):
                c = c1[t0 + i]
                s = s1[t0 + i]
                cn = ck[i] * c - sk[i] * s
                sk[i] = sk[i] * c + ck[i] * s
                ck[i] = cn
            k += 1
        for seg in range(n_seg):
            j = j0 + seg
            a_re = e_re[line, j]
            a_im = e_im[line, j]
            d_re = (e_re[line, j + 1] - a_re) * inv
            d_im = (e_im[line, j + 1] - a_im) * inv
            i0 = seg * hop_env
            i1 = min(i0 + hop_env, n)
            for i in range(i0, i1):
                fr = np.float32(i - i0)
                er = a_re + d_re * fr
                ei = a_im + d_im * fr
                zbuf[line, i] = er * ck[i] - ei * sk[i]
                zbuf[L + line, i] = er * sk[i] + ei * ck[i]


@njit(cache=True, fastmath=True)
def _ar1(g: np.ndarray, rho: np.ndarray, drive: np.ndarray) -> None:
    """``g[..., i] = rho g[..., i - 1] + drive[..., i]`` in place, from ``g[..., 0]``."""
    R, K, n = g.shape
    for r in range(R):
        for k in range(K):
            for i in range(1, n):
                g[r, k, i] = rho[r, k] * g[r, k, i - 1] + drive[r, k, i]


def _render_lines(
    audio: np.ndarray,
    c1: np.ndarray,
    s1: np.ndarray,
    env: np.ndarray,
    hop_env: int,
    mix: np.ndarray,
    orders: np.ndarray,
) -> None:
    """``audio (M, T) += Re(sum_l G_ml Z_l(t))`` for one rotor: ``env`` the
    ``(L, n_env)`` complex line envelopes of the ascending ``orders (L,)``,
    ``mix`` ``(M, L)`` complex mic gains ``g_ml e^{i alpha_ml}``."""
    _render_lines_ri(
        audio,
        c1,
        s1,
        np.ascontiguousarray(env.real, dtype=np.float32),
        np.ascontiguousarray(env.imag, dtype=np.float32),
        hop_env,
        mix,
        orders,
    )


def _render_lines_ri(
    audio: np.ndarray,
    c1: np.ndarray,
    s1: np.ndarray,
    e_re: np.ndarray,
    e_im: np.ndarray,
    hop_env: int,
    mix: np.ndarray,
    orders: np.ndarray,
    *,
    start: tuple[np.ndarray, np.ndarray, int] | None = None,
) -> None:
    """:func:`_render_lines` on the envelope's C-contiguous float32 ``Re`` and
    ``Im`` planes; the kernel's block is mixed by one BLAS product
    ``[Re G, -Im G] @ [Re Z; Im Z]``. ``start = (cos, sin)(k phi), k``: the
    harmonic the kernel's recurrence starts from (default ``phi`` itself), so a
    block of high orders does not walk the recurrence up from ``k = 1``."""
    L, T = e_re.shape[0], c1.shape[0]
    if L == 0:
        return
    c0, s0, k_init = (c1, s1, 1) if start is None else start
    a = np.concatenate([mix.real, -mix.imag], axis=1).astype(np.float32)
    k_l = np.ascontiguousarray(orders, dtype=np.int64)
    zbuf = np.empty((2 * L, LINE_BLOCK), dtype=np.float32)
    ck = np.empty(LINE_BLOCK, dtype=np.float32)
    sk = np.empty(LINE_BLOCK, dtype=np.float32)
    for t0 in range(0, T, LINE_BLOCK):
        n = min(LINE_BLOCK, T - t0)
        _line_kernel(c1, s1, c0, s0, int(k_init), e_re, e_im, hop_env, t0, n, zbuf, ck, sk, k_l)
        audio[:, t0 : t0 + n] += a @ zbuf[:, :n]


def _sc_wind(
    rng: np.random.Generator,
    spec: Mapping[str, Any],
    n_mics: int,
    n_work: int,
    sr_work: int,
    floor_band: np.ndarray,
    band: np.ndarray,
) -> np.ndarray:
    """``(n_mics, n_work)`` SC wind noise, the generator's own gusty wind-speed
    profile per capsule (``gustiness`` Weibull draws over the clip, SC's
    default 3). One clip-level intensity ``s`` (none with probability
    ``1 - p_wind``, else uniform), per capsule a mean level
    ``max_over_floor_db * s - Exp(shield_mean_db)`` dB over that capsule's
    20-100 Hz floor power."""
    from data_processing.noise_model.wind_sc import wind_noise

    out = np.zeros((n_mics, n_work))
    if rng.uniform() >= float(spec.get("p_wind", 0.7)):
        return out
    s = float(rng.uniform())
    for m in range(n_mics):
        excess_db = float(spec.get("max_over_floor_db", 33.0)) * s - rng.exponential(
            float(spec.get("shield_mean_db", 8.0))
        )
        if excess_db <= -10.0:
            continue
        w, _ = wind_noise(rng, sr_work, n_work / sr_work, gustiness=int(spec.get("gustiness", 3)))
        w = w[:n_work] if w.size >= n_work else np.pad(w, (0, n_work - w.size))
        w_band = float(np.mean(np.abs(np.fft.rfft(w)[band]) ** 2))
        out[m] = w * math.sqrt(floor_band[m] * 10.0 ** (excess_db / 10.0) / max(w_band, 1e-30))
    return out


def _power_mean_shift_db(var_db2: Any) -> np.ndarray:
    """The dB mean of a Gaussian of variance ``var_db2`` whose ``10^{x/10}``
    has mean one: ``-var ln10 / 20``."""
    return -np.asarray(var_db2, dtype=np.float64) * math.log(10.0) / 20.0


def _slow_gain(
    x: np.ndarray, gain_db: Any, *, sr_work: int, frame: int = WANDER_WOLA_FRAME
) -> np.ndarray:
    """``x`` with a slowly varying, frequency-dependent gain applied.

    ``gain_db(t_s) -> (len(t_s), frame // 2 + 1)`` gives the dB gain at
    frame-centre times ``t_s`` (seconds from the first sample) on the rfft
    bins of one frame. Weighted overlap-add with a sqrt-periodic-Hann pair at
    50 % overlap, which sums to one, so a constant gain returns ``x`` scaled
    exactly and a slow one multiplies the local power by ``10^{gain/10}``.
    """
    n = int(x.size)
    nf = int(frame)
    hop = nf // 2
    extra = (-(n + nf)) % hop
    xp = np.concatenate([np.zeros(nf), np.asarray(x, dtype=np.float64), np.zeros(nf + extra)])
    n_frames = (xp.size - nf) // hop + 1
    w = np.sqrt(0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(nf) / nf))
    idx = np.arange(n_frames)[:, None] * hop + np.arange(nf)[None, :]
    spec = np.fft.rfft(xp[idx] * w[None, :], axis=1)
    t_centre = (np.arange(n_frames) * hop + nf // 2 - nf) / float(sr_work)
    spec *= 10.0 ** (np.asarray(gain_db(t_centre), dtype=np.float64) / 20.0)
    frames = np.fft.irfft(spec, n=nf, axis=1) * w[None, :]
    halves = frames.reshape(n_frames, 2, hop)
    out = np.zeros((n_frames + 1, hop))
    out[:n_frames] += halves[:, 0]
    out[1:] += halves[:, 1]
    return out.reshape(-1)[nf : nf + n]


def fit_work_rate(fit: Mapping[str, Any]) -> int:
    """The work rate a fit renders at unless told otherwise: the one its
    likelihood's kernel ran on and whose render-chain transfer it carries
    (``front_end.sr_work``: 64 kHz for the v2 fits, 32 kHz for the GPU
    ``flight_v3`` fits), :data:`.spectrum.SAMPLE_RATE_WORK` for a v2/v3
    payload that records none (bench fits, hand-built payloads) and
    :data:`.v4.WORK_RATE_HZ` for a v4 one."""
    front_end = fit.get("front_end") or {}
    default = V4_WORK_RATE_HZ if fit.get("schema") == FIT_SCHEMA_V4 else SP.SAMPLE_RATE_WORK
    return int(front_end.get("sr_work") or default)


@overload
def render_noise(
    fit: dict[str, Any],
    rps_rev_s: np.ndarray,
    *,
    sr: int = SP.FLIGHT_SR,
    n_mics: int = 8,
    seed: int = 0,
    sr_work: int | None = None,
    wander_mean: WanderMean = "zero",
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
    sr_work: int | None = None,
    wander_mean: WanderMean = "zero",
    return_diagnostics: Literal[True],
) -> tuple[np.ndarray, dict[str, Any]]: ...


def render_noise(
    fit: dict[str, Any],
    rps_rev_s: np.ndarray,
    *,
    sr: int = SP.FLIGHT_SR,
    n_mics: int = 8,
    seed: int = 0,
    sr_work: int | None = None,
    wander_mean: WanderMean = "zero",
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Synthesise the fitted noise on the carriers ``rps_rev_s``.

    Parameters
    ----------
    fit
        A ``noise-v2-fit/{1,2}`` or ``noise-v3-fit/1`` payload
        (:func:`experiments.noise_model.fit.write_fit`'s output), or a
        ``noise-v4-fit/3`` one (:func:`_render_noise_v4`).
    rps_rev_s
        ``(R, T)`` per-rotor carrier in rev/s on the OUTPUT grid at ``sr``. A
        bench fit's own constant carriers are NOT used here: the renderer is
        driven by the carriers it is asked for, which is what lets a bench-fitted
        rig be rendered on a flight trajectory.
    sr, n_mics, seed
        Output rate, channel count and the single seed of every draw.
    sr_work
        The work rate the lines and the floor are synthesised at before the
        anti-alias and the decimation to ``sr``; ``None``: :func:`fit_work_rate`,
        the rate whose chain the fit's likelihood carried.
    wander_mean
        ``"zero"``: the v3 wander tracks drawn zero-mean in dB (the default);
        ``"power"``: the same draws shifted so every power multiplier has mean
        one (module docstring, WANDER MEAN). A v2 or v4 payload ignores it.

    Returns
    -------
    ``(n_mics, T)`` float64 in ABSOLUTE units — the same units the fitted ``M``
    and :func:`revised_eval.window_periodogram` are in. No RMS normalisation.
    """
    if wander_mean not in WANDER_MEANS:
        raise ValueError(f"wander_mean {wander_mean!r} is not one of {WANDER_MEANS}")
    sr_work = fit_work_rate(fit) if sr_work is None else int(sr_work)
    if fit.get("schema") == FIT_SCHEMA_V4:
        out_v4, diag_v4 = _render_noise_v4(
            fit, rps_rev_s, sr=int(sr), n_mics=int(n_mics), seed=int(seed), sr_work=sr_work
        )
        return (out_v4, diag_v4) if return_diagnostics else out_v4
    p = _check_schema(fit)
    v3 = fit.get("schema") == FIT_SCHEMA_V3
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
    gamma_hz = gamma_from_params(p)
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

    # every order the profile has that fits below the WORK Nyquist is rendered; the
    # ones above the OUTPUT Nyquist (or crossing it as the carrier moves) are then
    # removed by the same anti-alias chain a real recording's decimator applied -
    # a comb's top rolls off through that transition band, it never stops at the
    # last order under 8 kHz for the clip's fastest carrier (a cliff no recording has)
    k_max = min(
        int(profile_db.shape[1]),
        SP.k_max_for_carrier(f0.max(axis=1), sr_work, k_cap=int(profile_db.shape[1])),
    )
    if k_max < 1:
        raise ValueError(
            f"no order of a {float(f0.max()):.1f} rev/s rotor fits below {sr_work / 2} Hz"
        )

    ss = np.random.SeedSequence(int(seed))
    rng_state, rng_psi, rng_alpha, rng_floor = (np.random.default_rng(s) for s in ss.spawn(4))

    lam = float(p["lam"])
    sigma_nu = float(p["sigma_nu"])
    innov = rng_state.standard_normal((n_rotors, n_work, 2))
    theta, _nu = simulate_state(innov, lam=lam, sigma=sigma_nu, dt=dt)
    phase = 2.0 * np.pi * np.cumsum(f0, axis=1) * dt + theta  # (R, n_work)

    if v3:
        # no microphone block: the channels were normalised in the DATA
        mic_floor_db = np.zeros(n_mics)
        line_gain = np.ones((n_mics, n_rotors))
        all_gain = np.ones(n_mics)
    else:
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
        mic_line_db = (
            mic_line_db[:n_mics, :n_rotors] if mic_line_db.ndim == 2 else mic_line_db[:n_mics]
        )
        line_gain = 10.0 ** ((mic_line_db - mic_line_db.mean(axis=0, keepdims=True)) / 10.0)
        all_gain = 10.0 ** ((gain_all_db[:n_mics] - gain_all_db[:n_mics].mean()) / 10.0)

    # v3: FRESH block-wander tracks per clip, at the block centres, from
    # streams spawned after the four v2 ones (a v2 render draws nothing here)
    tracks: dict[str, np.ndarray] = {}
    wind_db: np.ndarray | None = None
    rng_wind: np.random.Generator | None = None
    t_knot = np.zeros(0)
    if v3:
        wander = Wander.from_mapping(p["wander"])
        rng_wander, rng_wind = (np.random.default_rng(s) for s in ss.spawn(2))
        n_blocks = max(1, int(math.ceil(n_out / float(sr) / wander.block_s)))
        t_knot = (np.arange(n_blocks) + 0.5) * wander.block_s
        n_ctrl = int(np.asarray(p["floor"]["floor_shape_z"]).size)
        for name, shape in (
            ("d", (n_rotors,)),
            ("v", (n_rotors, k_max)),
            ("u", ()),
            ("uj", (n_ctrl,)),
        ):
            tracks[name] = ou_blocks(
                rng_wander,
                shape,
                n_blocks,
                sigma=wander.track_sigma(name, k_max),
                rho=wander.track_rho(name, k_max) if wander.active(name) else 0.0,
            )
        mix = wander.uj_mix(SP.floor_ctrl_hz(sr))
        if mix is not None:
            # the colour moves smoothly across the control points (the
            # payload's ``uj_corr_oct``): each block's draw is ``A y``
            tracks["uj"] = mix @ tracks["uj"]
        if wander_mean == "power":
            # every family multiplies a power by 10^{x/10}: shift each track by
            # its own stationary variance (u_j through A y: diag(A A^T) sigma^2)
            uj_var = wander.sigma("uj") ** 2 * (
                np.ones(n_ctrl) if mix is None else np.einsum("ij,ij->i", mix, mix)
            )
            tracks["d"] += _power_mean_shift_db(wander.sigma("d") ** 2)
            v_sd = np.asarray(wander.track_sigma("v", k_max), dtype=np.float64)
            tracks["v"] += _power_mean_shift_db(v_sd**2)[..., None]
            tracks["u"] += _power_mean_shift_db(wander.sigma("u") ** 2)
            tracks["uj"] += _power_mean_shift_db(uj_var)[:, None]
        if p.get("wind") is not None:
            wind_db = np.asarray(p["wind"]["wind_db"], dtype=np.float64)
            if wind_db.size < n_mics:
                raise ValueError(f"fit carries {wind_db.size} wind levels, asked for {n_mics} mics")
    # PRIOR-RIG blocks (``experiments.noise_model.drone_prior``), all optional on
    # a v3 payload and drawn from streams spawned AFTER every existing one, so
    # a payload without them renders exactly as before:
    #   ``am``: per-(rotor, order) log-OU amplitude envelope shared by the mics
    #   ``profile.mic_dev_sd_db``: per-(mic, rotor, order) static dB deviation, redrawn per clip
    #   ``wind_sc``: SC wind noise (data_processing.noise_model.wind_sc) on some capsules
    mic_dev: np.ndarray | None = None
    rng_am = rng_dev = rng_wsc = None
    if v3 and (p.get("am") is not None or p["profile"].get("mic_dev_sd_db") or p.get("wind_sc")):
        rng_am, rng_dev, rng_wsc = (np.random.default_rng(s) for s in ss.spawn(3))
    dev_sd = float(p["profile"].get("mic_dev_sd_db") or 0.0)
    if v3 and dev_sd > 0.0:
        assert rng_dev is not None
        mic_dev = 10.0 ** (dev_sd * rng_dev.standard_normal((n_mics, n_rotors, k_max)) / 20.0)

    # LINES: every slow factor on the envelope grid (1 kHz), the carriers at the
    # rate each order needs. An order whose line stays under the anti-alias
    # passband edge for the clip's fastest carrier is rendered straight at the
    # output rate (the chain is unity there); the orders reaching the transition
    # band are rendered at the work rate and go through the chain
    # (``antialias`` + ``decimate_audio``), which is what shapes a comb's top.
    hop_env = int(sr) // ENV_RATE_HZ
    if hop_env * ENV_RATE_HZ != int(sr):
        raise ValueError(f"sr {sr} is not a multiple of the envelope rate {ENV_RATE_HZ}")
    dt_env = hop_env / float(sr)
    n_env = n_out // hop_env + 2
    t_env_s = np.arange(n_env) * dt_env
    knots = np.minimum(np.arange(n_env) * hop_env * oversample, n_work - 1)
    k_lo = min(k_max, SP.k_max_for_carrier(f0.max(axis=1), int(2 * AA_PASS_HZ), k_cap=k_max))
    am_env = None
    if v3 and p.get("am") is not None:
        am_s2 = np.asarray(p["am"]["sigma2"], dtype=np.float64)
        am_g = np.asarray(p["am"]["gamma_hz"], dtype=np.float64)
        if am_s2.shape[0] == 1:
            am_s2, am_g = np.repeat(am_s2, n_rotors, 0), np.repeat(am_g, n_rotors, 0)
        am_env = _am_envelopes(rng_am, am_s2[:, :k_max], am_g[:, :k_max], n_env, dt_env)
    amp_exp = float(p["profile"]["amp_exp"])
    speed = f0 / AMP_RPS_REF
    audio = np.zeros((n_mics, n_out), dtype=np.float32)
    audio_hi = np.zeros((n_mics, n_work), dtype=np.float32) if k_lo < k_max else None
    for r in range(n_rotors):
        env = (
            np.sqrt(2.0 * 10.0 ** (profile_db[r, :k_max] / 10.0))[:, None]
            * np.sqrt(speed[r, knots] ** amp_exp)[None, :]
        )  # (K, n_env)
        if v3:
            # the line's power wanders by 10^{(d_r + v_rk)/10}
            d_r = np.interp(t_env_s, t_knot, tracks["d"][r])
            v_r = np.stack([np.interp(t_env_s, t_knot, row) for row in tracks["v"][r]])
            env = env * 10.0 ** ((d_r[None, :] + v_r) / 20.0)
        if am_env is not None:
            env = env * am_env[r]
        # the line's own Wiener phase: increments N(0, 4 pi gamma dt_env),
        # cumulatively summed. Its increment variance IS the exponent the lag
        # law carries, so the rendered line is the Lorentzian of half-width
        # gamma_rk the fit was written in terms of.
        step = np.sqrt(4.0 * np.pi * np.maximum(gamma_hz[r, :k_max], 0.0) * dt_env)
        psi = np.cumsum(rng_psi.standard_normal((k_max, n_env)) * step[:, None], axis=1)
        env = env * np.exp(1j * psi)
        alpha = rng_alpha.uniform(0.0, 2.0 * np.pi, size=(n_mics, k_max))
        g = np.sqrt(line_gain[:, r] if line_gain.ndim == 2 else line_gain)[:, None] * np.ones(
            (1, k_max)
        )
        if mic_dev is not None:
            g = g * mic_dev[:, r, :]
        mix = g * np.exp(1j * alpha)
        ph = phase[r]
        ph_out = ph[::oversample]
        _render_lines(
            audio,
            np.cos(ph_out).astype(np.float32),
            np.sin(ph_out).astype(np.float32),
            env[:k_lo],
            hop_env,
            mix[:, :k_lo],
            np.arange(1, k_lo + 1),
        )
        if audio_hi is not None:
            # orders k_lo + 1 .. k_max at the work rate: the kernel walks the
            # angle-addition recurrence up to cos / sin((k_lo + 1) phi) first
            _render_lines(
                audio_hi,
                np.cos(ph).astype(np.float32),
                np.sin(ph).astype(np.float32),
                env[k_lo:],
                hop_env * oversample,
                mix[:, k_lo:],
                np.arange(k_lo + 1, k_max + 1),
            )
    audio = audio.astype(np.float64)
    if audio_hi is not None:
        audio += np.asarray(
            decimate_audio(antialias(audio_hi.astype(np.float64), sr_work), int(sr_work), int(sr)),
            dtype=np.float64,
        )[:, :n_out]
    # the broadband parts (floor, wind, array response): straight at the output rate
    n_work, dt, sr_work = n_out, 1.0 / float(sr), int(sr)
    speed, t_work = speed[:, ::oversample], t_work[::oversample]

    ctrl_hz = SP.floor_ctrl_hz(sr)
    shape_mat, tilt_oct = floor_geometry(np.fft.rfftfreq(n_work, d=dt), ctrl_hz)
    if v3:
        # the spline alone: c_j = mu + sigma_B (L z)_j, no tilt
        shape_db = SP.floor_shape_db(
            p["floor"]["floor_shape_z"], sr=sr, scale_db=float(p["floor"]["floor_shape_sd_db"])
        )
        tilt_db_oct = 0.0
    else:
        shape_db = SP.floor_shape_db(p["floor"]["floor_shape_z"], sr=sr)
        tilt_db_oct = float(p["floor"]["floor_tilt_db_oct"])
    floor_psd = floor_power_spectrum(
        shape_mat,
        tilt_oct,
        mean_db=float(p["floor"]["floor_mean_db"]),
        ctrl_db=shape_db,
        tilt_db_oct=tilt_db_oct,
        rate_factor=1.0,
    )
    floor_gain_t = (speed ** float(p["floor"]["floor_exp"])).mean(axis=0) + float(
        p["floor"]["floor_static_rel"]
    )
    floor_amp = np.sqrt(floor_psd)
    floor_moves = v3 and bool(np.any(tracks["u"]) or np.any(tracks["uj"]))
    wola_basis = (
        floor_geometry(np.fft.rfftfreq(WANDER_WOLA_FRAME, d=dt), ctrl_hz)[0]
        if floor_moves
        else None
    )

    def floor_wander_db(t_s: np.ndarray) -> np.ndarray:
        """``u(t) + sum_j B_j(f) u_j(t)`` on the WOLA frame's bins."""
        assert wola_basis is not None
        level = np.interp(t_s, t_knot, tracks["u"])
        colour = np.stack([np.interp(t_s, t_knot, row) for row in tracks["uj"]])  # (J, n_t)
        return level[:, None] + colour.T @ wola_basis.T

    floor_band = np.zeros(
        n_mics
    )  # each capsule's floor power in 20-100 Hz, the SC wind's reference
    band = (np.fft.rfftfreq(n_work, d=dt) >= 20.0) & (np.fft.rfftfreq(n_work, d=dt) <= 100.0)
    for m in range(n_mics):
        white = rng_floor.standard_normal(n_work)
        shaped = np.fft.irfft(np.fft.rfft(white) * floor_amp, n=n_work)
        if floor_moves:
            shaped = _slow_gain(shaped, floor_wander_db, sr_work=int(sr_work))
        sig = shaped * np.sqrt(floor_gain_t) * math.sqrt(10.0 ** (mic_floor_db[m] / 10.0))
        audio[m] += sig
        if v3 and p.get("wind_sc") is not None:
            floor_band[m] = float(np.mean(np.abs(np.fft.rfft(sig)[band]) ** 2))
    if v3 and p.get("wind_sc") is not None:
        assert rng_wsc is not None
        audio += _sc_wind(rng_wsc, p["wind_sc"], n_mics, n_work, int(sr_work), floor_band, band)
    if wind_db is not None:
        assert rng_wind is not None
        # static, per capsule: no speed law, no channel gain, no wander
        wind_amp = np.sqrt(wind_shape(np.fft.rfftfreq(n_work, d=dt)))
        for m in range(n_mics):
            white = rng_wind.standard_normal(n_work)
            level = math.sqrt(10.0 ** (float(wind_db[m]) / 10.0))
            audio[m] += level * np.fft.irfft(np.fft.rfft(white) * wind_amp, n=n_work)
    audio *= np.sqrt(all_gain)[:, None]
    response = p.get("array_response") if v3 else None
    if response is not None:
        # the ARRAY's measured per-channel response (round 4): the data were
        # normalised by it, so mic m is rendered back through 10^{g_m(f)/20}
        curve = array_response_db(response, np.fft.rfftfreq(n_work, d=dt))
        if curve.shape[0] < n_mics:
            raise ValueError(
                f"array_response carries {curve.shape[0]} channels, the render asks for {n_mics}"
            )
        for m in range(n_mics):
            audio[m] = np.fft.irfft(np.fft.rfft(audio[m]) * 10.0 ** (curve[m] / 20.0), n=n_work)

    # the chain's low-pass at the output rate: a recording's floor and its top
    # orders roll off through the decimator's transition band (the orders rendered
    # at the work rate have been through it once already; the band is 100 Hz wide)
    rendered = antialias(audio, sr)[:, :n_out]
    if not return_diagnostics:
        return rendered
    diagnostics: dict[str, Any] = dict(
        seed=int(seed),
        sr=int(sr),
        sr_work=int(sr_work),
        n_orders=int(k_max),
        n_rotors=int(n_rotors),
        n_mics=int(n_mics),
        rms=np.sqrt((rendered**2).mean(axis=1)).tolist(),
        normalisation="none (absolute fitted level)",
    )
    if v3:
        # the FRESH tracks this clip was drawn with, at the block centres: what
        # a prior-predictive or a recovery check compares the fit against
        diagnostics["wander_tracks"] = dict(t_knot_s=t_knot, **tracks)
    return rendered, diagnostics


# ── noise model v4 ──────────────────────────────────────────────────────────


def _render_noise_v4(
    fit: Mapping[str, Any],
    rps_rev_s: np.ndarray,
    *,
    sr: int,
    n_mics: int,
    seed: int,
    sr_work: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """A ``noise-v4-fit/3`` payload on the carriers ``rps_rev_s``
    (``docs/explainers/noise-model-v4.qmd`` § The model, § The renderer).

    WHAT IS DRAWN, fresh per clip from six streams of ``seed``:

    * per rotor, ONE integrated-OU shaft path at the work rate
      (:func:`.ou.simulate_state`), shared by the rotor's orders and mics, with
      ``(sigma_nu, lam)`` from the shaft law at the rotor's CLIP-MEAN speed (the
      transition is time-invariant; the flight law is flat);
    * per aerodynamic line ``(r, k)``, on the 1 kHz envelope grid, the complex
      envelope ``sqrt(2 P_rk(t)) exp(a - sigma2_amp) exp(i psi)``:
      ``P_rk(t) = 10^{p/10} s_r(t)^alpha_a``, ``a`` a stationary OU of
      variance ``sigma2_amp`` and rate ``gamma_amp_hz`` in log-amplitude, ``psi``
      an independent stationary OU of variance ``sigma2_phase`` and rate
      ``gamma_phase_hz`` ON THE PHASE (bounded, not a Wiener phase). The
      ``- sigma2_amp`` keeps ``10^{p/10}`` the MEAN power, as the lag law (one
      at ``tau = 0``) says it is (:mod:`.v4` module docstring);
    * per motor line ``(r, k = 3 p n)``: ``sqrt(2 P^m_rn(t))`` on the same
      carrier, no pedestal;
    * per (rotor, mic, line) a fixed uniform start phase, and the mic gain
      ``10^{E_rm/20}`` of the ROTOR-SPECIFIC response read at the line's
      CLIP-MEAN frequency ``k mean(f_r)`` — the bench pair EQ, or the flight
      ``Delta G_rm`` (frequency-independent, so exact). The flight array
      response ``A_m(f)`` is applied to the whole channel at the end, exactly.
      On the fitted trajectories (``data/rps-traj-fits``, michaels + dregon) the
      R3 bench pair EQ read at the clip-mean instead of the instantaneous
      frequency is off by 0.12 dB at the 95th percentile and 0.4 dB at the
      99th over 2 s clips (0.18 / 1.0 dB over 4 s); the clip's mean line
      power by 0.02 dB at the 95th;
    * the floor per mic: white noise shaped by ``sqrt(F(f))`` and given the
      short-time gain ``sum_r 10^{E_rm(f)/10} s_r(t)^{b(f)} + s`` per bin
      through the WOLA of :func:`_slow_gain` — the SUM of the rotors'
      independent Gaussian floors, drawn as one Gaussian with their summed
      spectrum; the static share ``s`` carries no rotor response;
    * flight only: the static per-mic wind block (v3's ``wind_db``), then the
      array response.

    The lines are synthesised by :func:`_line_kernel` on the 1 kHz envelope
    knots, the orders under the anti-alias passband edge at the output rate and
    the rest at ``sr_work`` through the chain, as v3. The kernel's chord between
    two knots keeps ``2/3 + rho/3`` of a line's power (``rho`` the envelope's
    correlation over one knot: 0.97 at a 5 Hz, 1 rad^2 phase OU, 0.86 at
    25 Hz); each aerodynamic envelope is raised by the inverse, so the
    rendered MEAN power is the payload's.
    """
    par = V4Params.from_payload(fit)
    rps = np.atleast_2d(np.asarray(rps_rev_s, dtype=np.float64))
    n_rotors, n_out = rps.shape
    if n_rotors != par.n_rotors:
        raise ValueError(f"fit carries {par.n_rotors} rotors, asked to render {n_rotors}")
    if n_mics > par.n_mics:
        raise ValueError(f"fit carries {par.n_mics} microphones, asked for {n_mics}")
    oversample = sr_work // sr
    if oversample * sr != sr_work:
        raise ValueError(f"sr_work {sr_work} must be an integer multiple of sr {sr}")
    hop_env = sr // ENV_RATE_HZ
    if hop_env * ENV_RATE_HZ != sr:
        raise ValueError(f"sr {sr} is not a multiple of the envelope rate {ENV_RATE_HZ}")
    n_work = n_out * oversample
    dt = 1.0 / float(sr_work)
    t_src = np.arange(n_out) / float(sr)
    t_work = np.arange(n_work) / float(sr_work)
    f0 = np.maximum(np.stack([np.interp(t_work, t_src, r) for r in rps]), SPEED_FLOOR_RPS)
    k_max = min(par.k_max, SP.k_max_for_carrier(f0.max(axis=1), sr_work, k_cap=par.k_max))
    k_lo = min(k_max, SP.k_max_for_carrier(f0.max(axis=1), int(2 * AA_PASS_HZ), k_cap=k_max))
    rng_state, rng_alpha, rng_amp, rng_phase, rng_floor, rng_wind = (
        np.random.default_rng(s) for s in np.random.SeedSequence(int(seed)).spawn(6)
    )

    f_bar = f0.mean(axis=1)
    s_bar = par.speed(f_bar)
    sigma_nu, lam = par.shaft.sigma_nu(s_bar), par.shaft.lam(s_bar)
    innov = rng_state.standard_normal((n_rotors, n_work, 2))
    theta = np.stack(
        [
            simulate_state(innov[r], lam=float(lam[r]), sigma=float(sigma_nu[r]), dt=dt)[0]
            for r in range(n_rotors)
        ]
    )
    phase = 2.0 * np.pi * np.cumsum(f0, axis=1) * dt + theta  # (R, n_work)

    # LINES: every slow factor on the 1 kHz envelope grid, in the kernel's
    # float32, as the Re / Im planes of the lines sorted by order
    dt_env = hop_env / float(sr)
    n_env = n_out // hop_env + 2
    knots = np.minimum(np.arange(n_env) * hop_env * oversample, n_work - 1)
    p_aero, p_motor = par.line_power(par.speed(f0[:, knots]))  # (R, K, n_env), (R, n_m, n_env)
    ped = par.pedestal
    block = (n_rotors, k_max)
    a = _ou_paths(
        rng_amp,
        np.broadcast_to(ped.sigma2_amp[:k_max], block),
        np.broadcast_to(ped.gamma_amp_hz[:k_max], block),
        n_env,
        dt_env,
        np.float32,
    )
    psi = _ou_paths(
        rng_phase,
        np.broadcast_to(ped.sigma2_phase[:k_max], block),
        np.broadcast_to(ped.gamma_phase_hz[:k_max], block),
        n_env,
        dt_env,
        np.float32,
    )
    a -= ped.sigma2_amp[:k_max, None].astype(np.float32)
    # the kernel's chord between two knots keeps 2/3 + rho/3 of a line's power,
    # rho = exp(s2A (e^{-2 pi gA dt} - 1) + s2P (e^{-2 pi gP dt} - 1)) the
    # envelope's correlation over one knot: each aerodynamic envelope is raised
    # by the inverse, so the rendered mean power stays the payload's
    rho_knot = np.exp(
        ped.sigma2_amp[:k_max] * np.expm1(-2.0 * np.pi * ped.gamma_amp_hz[:k_max] * dt_env)
        + ped.sigma2_phase[:k_max] * np.expm1(-2.0 * np.pi * ped.gamma_phase_hz[:k_max] * dt_env)
    )
    chord = 1.0 / np.sqrt(2.0 / 3.0 + rho_knot / 3.0)  # (K,)
    mod_aero = (np.sqrt(2.0 * p_aero[:, :k_max]) * chord[None, :, None]).astype(np.float32)
    mod_aero *= np.exp(a)  # (R, K, n_env)
    motor = par.motor_orders <= k_max
    mod_motor = np.sqrt(2.0 * p_motor[:, motor]).astype(np.float32)
    orders = np.concatenate([np.arange(1, k_max + 1), par.motor_orders[motor]])
    by_order = np.argsort(orders, kind="stable")
    row = np.empty_like(by_order)
    row[by_order] = np.arange(by_order.size)  # each line's row in order
    orders = orders[by_order]
    n_lo = int(np.searchsorted(orders, k_lo, side="right"))  # rows rendered at sr
    e_re = np.zeros((orders.size, n_env), dtype=np.float32)
    e_im = np.zeros_like(e_re)  # the motor rows stay real
    audio = np.zeros((n_mics, n_out), dtype=np.float32)
    audio_hi = np.zeros((n_mics, n_work), dtype=np.float32) if n_lo < orders.size else None
    for r in range(n_rotors):
        e_re[row[:k_max]] = mod_aero[r] * np.cos(psi[r])
        e_im[row[:k_max]] = mod_aero[r] * np.sin(psi[r])
        e_re[row[k_max:]] = mod_motor[r]
        gain_db = par.mic.rotor_db(orders * f_bar[r])[r, :n_mics]  # (M, L)
        alpha = rng_alpha.uniform(0.0, 2.0 * np.pi, size=gain_db.shape)
        mix = 10.0 ** (gain_db / 20.0) * np.exp(1j * alpha)
        ph = phase[r]
        ph_out = ph[::oversample]
        _render_lines_ri(
            audio,
            np.cos(ph_out).astype(np.float32),
            np.sin(ph_out).astype(np.float32),
            e_re[:n_lo],
            e_im[:n_lo],
            hop_env,
            mix[:, :n_lo],
            orders[:n_lo],
        )
        if audio_hi is not None:
            k_hi = int(orders[n_lo])
            _render_lines_ri(
                audio_hi,
                np.cos(ph).astype(np.float32),
                np.sin(ph).astype(np.float32),
                e_re[n_lo:],
                e_im[n_lo:],
                hop_env * oversample,
                mix[:, n_lo:],
                orders[n_lo:],
                start=(
                    np.cos(k_hi * ph).astype(np.float32),
                    np.sin(k_hi * ph).astype(np.float32),
                    k_hi,
                ),
            )
    out = audio.astype(np.float64)
    if audio_hi is not None:
        out += np.asarray(
            decimate_audio(antialias(audio_hi.astype(np.float64), sr_work), sr_work, sr),
            dtype=np.float64,
        )[:, :n_out]

    # FLOOR (and wind, array response): straight at the output rate
    freqs = np.fft.rfftfreq(n_out, d=1.0 / float(sr))
    floor_amp = np.sqrt(par.floor.psd(freqs))
    f_wola = np.fft.rfftfreq(V4_FLOOR_WOLA_FRAME, d=1.0 / float(sr))
    b_wola = par.floor.exponent(f_wola)
    rotor_lin = 10.0 ** (par.mic.rotor_db(f_wola)[:, :n_mics] / 10.0)  # (R, M, Fw)
    t_out = np.arange(n_out) / float(sr)
    log_s = np.log(par.speed(f0[:, ::oversample]))
    speed_law: dict[int, np.ndarray] = {}

    def rotor_floor_gain(t_s: np.ndarray) -> np.ndarray:
        """``(R, n_t, Fw)`` ``s_r(t)^{b(f)}`` at the WOLA frame centres (the
        same centres for every mic, so computed once)."""
        if t_s.size not in speed_law:
            ls = np.stack([np.interp(t_s, t_out, row) for row in log_s])
            speed_law[t_s.size] = np.exp(ls[:, :, None] * b_wola[None, None, :])
        return speed_law[t_s.size]

    static = par.floor.static_rel
    for m in range(n_mics):
        white = rng_floor.standard_normal(n_out)
        shaped = np.fft.irfft(np.fft.rfft(white) * floor_amp, n=n_out)

        def floor_gain_db(t_s: np.ndarray, m: int = m) -> np.ndarray:
            g = np.einsum("rf,rtf->tf", rotor_lin[:, m], rotor_floor_gain(t_s)) + static
            return 10.0 * np.log10(g)

        out[m] += _slow_gain(shaped, floor_gain_db, sr_work=sr, frame=V4_FLOOR_WOLA_FRAME)
    if par.wind_db is not None:
        wind_amp = np.sqrt(wind_shape(freqs))
        for m in range(n_mics):
            white = rng_wind.standard_normal(n_out)
            level = math.sqrt(10.0 ** (float(par.wind_db[m]) / 10.0))
            out[m] += level * np.fft.irfft(np.fft.rfft(white) * wind_amp, n=n_out)
    if par.mic.array_response is not None:
        curve = par.mic.array_db(freqs)
        for m in range(n_mics):
            out[m] = np.fft.irfft(np.fft.rfft(out[m]) * 10.0 ** (curve[m] / 20.0), n=n_out)
    rendered = antialias(out, sr)[:, :n_out]
    diagnostics: dict[str, Any] = dict(
        schema=FIT_SCHEMA_V4,
        seed=int(seed),
        sr=int(sr),
        sr_work=int(sr_work),
        n_orders=int(k_max),
        n_rotors=int(n_rotors),
        n_mics=int(n_mics),
        shaft=dict(sigma_nu_rad_s=sigma_nu.tolist(), lam_per_s=lam.tolist()),
        rms=np.sqrt((rendered**2).mean(axis=1)).tolist(),
        normalisation="none (absolute fitted level)",
    )
    return rendered, diagnostics


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
    sr_work: int | None = None,
    wander_mean: WanderMean = "zero",
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
    the band costs two. ``wander_mean`` goes to each :func:`render_noise`.
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
                wander_mean=wander_mean,
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
        sr_work={r: fit_work_rate(fits[r]) if sr_work is None else int(sr_work) for r in fits},
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
