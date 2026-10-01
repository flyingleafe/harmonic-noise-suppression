"""Per-rotor IDENTIFIABILITY gate of a v3 rig payload on a trajectory pool.

The rule (agreed with Dmitrii, 2026-09-29; ``docs/experiments/noise-model-v3.md``
§ "Identifiability gate"):

* A harmonic ``k`` of rotor ``r`` is IDENTIFIABLE in a frame when its line
  raises the frame's mic-mean expected spectrum at the line centre (the bin
  nearest ``k f_r``) by at least ``threshold_db`` over everything else there:
  ``10 log10((L + B) / B) >= T``. Any order in ``[f_min_hz, f_max_hz]`` counts;
  a lone line (DREGON's order 70) counts.
* A rotor PASSES a frame when at least ``min_orders`` of its harmonics are
  identifiable there.
* A rotor is EXCUSED in a frame that no rig can pass: stopped
  (``spin_min_rps``) or closer than ``sep_rps`` to another rotor anywhere in
  the frame. The distance is geometry only, never the rig's line width.
* A rig PASSES when every rotor passes at least ``pass_frac`` of its eligible
  frames AND has at least ``min_eligible_frames`` of them: a pool that rarely
  separates a rotor makes it UNTESTABLE (:func:`pool_coverage`, a property of
  the pool), which fails - the gate is never passed vacuously.

HOW ``L`` AND ``B`` ARE COMPUTED (:func:`line_contrasts`, Dmitrii's scheme):

* ``L``: the line's expected periodogram at constant speed (the frame-centre
  speed). At constant speed the forward model's line is a SHIFTED SHAPE: its
  atom is ``A s w(t) e^{i 2 pi k f t}``, so the kernel of
  :func:`spectrum.flight_line_spectra` reduces exactly to
  ``1/2 10^{p/10} (f / f_ref)^a (sr / sr_work) / sum(w^2) [H(nu - k f) + H(nu + k f)]``
  with ``H_rk(d) = sum_tau A_w(tau) rho_rk(tau) cos(2 pi d tau / sr_work)``,
  ``A_w`` the work window's autocorrelation and ``rho_rk`` the fitted lag law
  (:func:`lag.r_tau`: width ``gamma_rk``, shaft ``sigma_nu``, ``lam``), which
  does not depend on the speed. ``H`` is computed ONCE per rig on a fine
  offset grid (:func:`line_shapes`); a frame only reads it.
* ``B``: the model's floor (and DREGON's wind) at that bin
  (:func:`spectrum.floor_unit_autocorr` / ``floor_frames_from_autocov``, with
  the frame's real speed track), plus EVERY other line of every rotor at that
  bin through the same shapes (tabulated over the whole band). The one
  simplification is the constant speed within a frame (Dmitrii's call,
  2026-09-29): on real stream frames it reads contrasts higher than the exact
  model (``docs/experiments/noise-model-v3.md`` § "Identifiability gate").
* Mics: every quantity is the MEAN over mics of the observed periodogram: the
  mic gains, the transfer and the array response (round 4) are applied per mic
  before the mean. Regimes: power-blended per frame by
  :func:`regime_blend_weight`, as :func:`render.expected_periodogram_regimes`.

:func:`decompose` is the EXACT reference (every line of every rotor through
:func:`spectrum.flight_cache`, chirp included); the tests pin the fast scheme
against it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from data_processing.noise_model.constants import AMP_RPS_REF, SPEED_FLOOR_RPS
from data_processing.noise_model.params import array_response_db, check_schema
from data_processing.noise_model.render import fit_work_rate, regime_blend_weight

from . import model as MD
from . import spectrum as SP
from .lag import r_tau

__all__ = [
    "Coverage",
    "Decomposition",
    "FramePool",
    "GateConfig",
    "GateResult",
    "LineShapes",
    "decompose",
    "frame_pool",
    "gate",
    "line_contrasts",
    "line_shapes",
    "pool_coverage",
    "regimes_of",
]


@dataclass(frozen=True)
class GateConfig:
    """Every number the gate reads. Defaults: the agreed rule."""

    #: ``T``: dB the line must add to everything else at its centre bin.
    threshold_db: float = 2.0
    #: ``D``: a rotor closer than this to another (rev/s, anywhere in the frame) is excused.
    sep_rps: float = 1.5
    #: A rotor passes a frame with at least this many identifiable harmonics.
    min_orders: int = 3
    #: A rig passes when every rotor passes this fraction of its eligible frames.
    pass_frac: float = 0.9
    #: ... and is eligible in at least this many frames of the pool (else untestable).
    min_eligible_frames: int = 10
    #: A rotor at or below this speed (rev/s, anywhere in the frame) is stopped.
    spin_min_rps: float = 0.0
    f_min_hz: float = SP.BAND_F_MIN
    f_max_hz: float = SP.BAND_F_MAX
    #: Largest spacing (Hz) of the offset grid the line shapes are tabulated on.
    shape_step_hz: float = 0.5


@dataclass(frozen=True)
class FramePool:
    """Frames drawn from trajectory windows: ``windows[i]`` is ``(R, T)`` rev/s at
    ``sr``, ``starts[i]`` the sample starts of its frames kept in the pool,
    framed at ``n_fft`` / ``hop``."""

    windows: tuple[np.ndarray, ...]
    starts: tuple[np.ndarray, ...]
    seed: int
    n_fft: int = SP.FLIGHT_N_FFT
    hop: int = SP.FLIGHT_HOP
    sr: int = SP.FLIGHT_SR

    @property
    def n_frames(self) -> int:
        return int(sum(s.size for s in self.starts))

    @property
    def n_rotors(self) -> int:
        return int(self.windows[0].shape[0])


def frame_pool(
    windows: Sequence[np.ndarray],
    *,
    n_frames: int,
    seed: int,
    n_fft: int = SP.FLIGHT_N_FFT,
    hop: int = SP.FLIGHT_HOP,
    sr: int = SP.FLIGHT_SR,
) -> FramePool:
    """``n_frames`` frames drawn uniformly without replacement from every
    STFT frame (``data.periodogram``'s framing) of ``windows`` (rev/s at ``sr``)."""
    wins = tuple(np.atleast_2d(np.asarray(w, dtype=np.float64)) for w in windows)
    if not wins:
        raise ValueError("a frame pool needs at least one trajectory window")
    if len({w.shape[0] for w in wins}) != 1:
        raise ValueError("every window must carry the same rotors")
    per = [np.arange(1 + (w.shape[1] - int(n_fft)) // int(hop)) * int(hop) for w in wins]
    if any(p.size == 0 for p in per):
        raise ValueError(f"a window is shorter than n_fft {n_fft}")
    owner = np.concatenate([np.full(p.size, i) for i, p in enumerate(per)])
    flat = np.concatenate(per)
    take = min(int(n_frames), flat.size)
    pick = np.sort(np.random.default_rng(int(seed)).choice(flat.size, size=take, replace=False))
    starts = tuple(np.sort(flat[pick][owner[pick] == i]) for i in range(len(wins)))
    return FramePool(
        windows=wins, starts=starts, seed=int(seed), n_fft=int(n_fft), hop=int(hop), sr=int(sr)
    )


def regimes_of(fits: Mapping[str, dict[str, Any]] | dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``{"single": fit}`` for one payload, or the ``standby``/``cruise`` pair."""
    if "params" in fits:
        return {"single": dict(fits)}
    keys = set(fits)
    if keys == {"single"} or keys == {"standby", "cruise"}:
        return {str(k): dict(v) for k, v in fits.items()}
    raise ValueError(f"expected a payload, 'single', or 'standby'+'cruise'; got {sorted(keys)}")


def _regime_weights(
    regimes: Mapping[str, dict[str, Any]], rps: np.ndarray, centres: np.ndarray, sr: int
) -> dict[str, np.ndarray]:
    """Each regime's power weight at the frame centres."""
    if "single" in regimes:
        return {"single": np.ones(centres.size)}
    w = regime_blend_weight(rps, sr=sr)[centres]
    return {"standby": 1.0 - w, "cruise": w}


# -- one regime payload, on the model's grid -----------------------------------


@dataclass
class _Payload:
    """One regime payload on the flight grid, with its per-mic observed factors
    at the analysis bins: ``line_obs`` ``(R, F)`` is the mic mean of
    ``all_gain_m * array_m(f) * line_gain_mr`` times the transfer."""

    grid: SP.FlightGrid
    params: SP.V2Params
    k_prof: int
    array: torch.Tensor  # (M, F): the array response, linear (ones without one)
    mic_obs: torch.Tensor  # (M, F): all_gain_m * array_m(f) * transfer(f)
    line_obs: torch.Tensor  # (R, F)
    #: ``(sigma2, gamma_hz)``, each ``(R, K)``: the prior rigs' per-order log-OU
    #: amplitude envelope (``render._am_envelopes``); ``None`` on a fit.
    am: tuple[np.ndarray, np.ndarray] | None = None


def _payload(fit: dict[str, Any], *, n_fft: int, hop: int, sr: int) -> _Payload:
    p = check_schema(fit)
    grid = SP.flight_grid(sr=sr, n_fft=n_fft, hop=hop, sr_work=fit_work_rate(fit))
    freqs = np.fft.rfftfreq(int(n_fft), 1.0 / sr)
    response = p.get("array_response")
    curve = None if response is None else array_response_db(response, freqs)
    params = MD.params_from_dict(p, n_mics=None if curve is None else int(curve.shape[0]))
    ref = grid.window_work
    line_gain = SP._mean_pinned_db(torch.as_tensor(params.mic_line_gain_db, dtype=ref.dtype))
    all_gain = SP._mean_pinned_db(
        torch.as_tensor(params.gain_all_db, dtype=ref.dtype).reshape(-1, 1)
    ).reshape(-1)
    n_mics = int(line_gain.shape[0])
    array = torch.ones(n_mics, freqs.size, dtype=ref.dtype)
    if curve is not None:
        if curve.shape[0] != n_mics:
            raise ValueError(f"array response has {curve.shape[0]} channels, payload {n_mics}")
        array = torch.as_tensor(10.0 ** (curve / 10.0), dtype=ref.dtype)
    mic_obs = all_gain[:, None] * array * grid.transfer_power[None, :]
    line_obs = torch.einsum("mf,mr->rf", mic_obs, line_gain) / n_mics
    am = p.get("am")
    return _Payload(
        grid=grid,
        params=params,
        k_prof=int(np.asarray(p["profile"]["profile_db"]).shape[1]),
        array=array,
        mic_obs=mic_obs,
        line_obs=line_obs,
        am=None
        if am is None
        else (
            np.asarray(am["sigma2"], dtype=np.float64),
            np.asarray(am["gamma_hz"], dtype=np.float64),
        ),
    )


def _floor_obs(pl: _Payload, rps: np.ndarray, starts: np.ndarray) -> torch.Tensor:
    """``(n, F)`` mic mean of the observed floor (and wind) of each frame, on
    the frame's real speed track (:func:`spectrum.flight_model_cached`'s floor)."""
    grid, params = pl.grid, pl.params
    rate = SP.flight_rate_work(grid, rps, starts)
    g1 = SP.floor_unit_autocorr(grid, rate, params.floor)
    c = grid.floor.autocovariance_db(grid.floor.log_psd_db(params.floor))[None, :]
    floor = SP.floor_frames_from_autocov(grid, g1, c)  # (n, F)
    mic_floor = 10.0 ** (torch.as_tensor(params.floor.mic_floor_db, dtype=floor.dtype) / 10.0)
    per_mic = floor[None] * mic_floor.reshape(-1, 1, 1)
    if params.wind_db is not None:
        per_mic = per_mic + SP.wind_frames(grid, params)[:, None, :]
    return (per_mic * pl.mic_obs[:, None, :]).mean(dim=0)


# -- the line shapes ------------------------------------------------------------


@dataclass
class LineShapes:
    """``H_rk(d)`` of one payload on the offsets ``d = i * step_hz``
    (``i = 0 .. M - 1``, up to half the work rate; ``H`` is even), scaled so
    that the line's observed mic-mean power at bin ``nu`` is
    ``level[r, k] * (f / f_ref)^amp_exp * line_obs[r, nu] * (H(nu - k f) + H(nu + k f))``."""

    shape: np.ndarray  # (R, K, M)
    level: np.ndarray  # (R, K): 1/2 10^{p/10} (sr / sr_work) / sum(w^2)
    amp_exp: np.ndarray  # (R,) or scalar broadcast
    step_hz: float

    def speed_law(self, f_rps: np.ndarray) -> np.ndarray:
        """``(R, n)`` amplitude speed law of every rotor at speeds ``f_rps`` ``(R, n)``."""
        a = np.broadcast_to(self.amp_exp, (self.shape.shape[0],))
        return (np.maximum(f_rps, SPEED_FLOOR_RPS) / AMP_RPS_REF) ** a[:, None]


def line_shapes(pl: _Payload, *, step_hz: float) -> LineShapes:
    """Every line's ``H_rk`` of payload ``pl`` on offsets ``0 .. sr_work / 2`` at
    a spacing of at most ``step_hz``: the lag sequence ``A_w(tau) rho_rk(tau)``,
    ``tau = 0 .. n - 1``, zero-padded and transformed once per line
    (``H = 2 Re rfft - x_0``, the even lag sum). A prior rig's amplitude
    envelope multiplies the lag law by its own autocorrelation
    ``exp(sigma2 e^{-2 pi gamma |tau|})`` (unit mean amplitude: ``1`` at long
    lags, the pedestal's ``e^{sigma2} - 1`` at zero lag)."""
    grid, params = pl.grid, pl.params
    n = int(grid.n_fft_work)
    sr_work = int(grid.sr_work)
    big = 1 << max(math.ceil(math.log2(sr_work / float(step_hz))), math.ceil(math.log2(2 * n)))
    w = grid.window_work
    spec = torch.fft.fft(w.to(torch.complex128), n=2 * n)
    aw = torch.fft.ifft(spec.real**2 + spec.imag**2).real[:n]  # window autocorrelation
    n_rot = int(torch.as_tensor(params.profile_db).shape[0])
    gamma = SP.gamma_block(params.gamma_hz, n_rotors=n_rot, k_max=pl.k_prof, ref=w)
    k = torch.arange(1, pl.k_prof + 1, dtype=torch.float64)
    out = np.empty((n_rot, pl.k_prof, big // 2 + 1))
    with torch.no_grad():
        for k0 in range(0, pl.k_prof, 16):
            k1 = min(k0 + 16, pl.k_prof)
            rho = r_tau(
                grid.tau_s_work[None, None, :],
                k[None, k0:k1, None],
                sigma_nu=params.sigma_nu,
                lam=params.lam,
                gamma_hz=gamma[:, k0:k1, None],
            )  # (R, k, n)
            x = aw * rho
            if pl.am is not None:
                s2 = torch.as_tensor(pl.am[0][:, k0:k1, None], dtype=x.dtype)
                g = torch.as_tensor(pl.am[1][:, k0:k1, None], dtype=x.dtype)
                x = x * torch.exp(
                    s2 * torch.exp(-2.0 * math.pi * g * grid.tau_s_work[None, None, :])
                )
            out[:, k0:k1] = (2.0 * torch.fft.rfft(x, n=big).real - x[..., :1]).numpy()
    prof = torch.as_tensor(params.profile_db, dtype=torch.float64)
    level = 0.5 * 10.0 ** (prof / 10.0) * grid.grid_power_factor / float(grid.window_work_sumsq)
    return LineShapes(
        shape=out,
        level=level.numpy(),
        amp_exp=np.asarray(torch.as_tensor(params.amp_exp).detach().numpy(), dtype=np.float64),
        step_hz=sr_work / big,
    )


def _shape_at(flat: np.ndarray, q: np.ndarray, d_hz: np.ndarray, step_hz: float) -> np.ndarray:
    """``H_q(d)`` of line ``q`` (rows of ``flat``) by linear interpolation."""
    x = np.abs(d_hz) / step_hz
    i0 = x.astype(np.int64)
    if int(i0.max()) > flat.shape[1] - 2:
        raise ValueError(f"offset {float(np.abs(d_hz).max()):.1f} Hz is past the shape table")
    frac = x - i0
    return flat[q, i0] * (1.0 - frac) + flat[q, i0 + 1] * frac


def line_contrasts(
    fits: Mapping[str, dict[str, Any]] | dict[str, Any],
    pool: FramePool,
    config: GateConfig | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per window of ``pool``: ``(contrast_db, centre_hz)``, each ``(R, n, K)``:
    ``10 log10((L + B) / B)`` of every line ``(r, k)`` at its centre bin, and
    that centre ``k f_r``. ``B`` sums the floor and EVERY other line of every
    rotor at that bin. ``K`` is the largest profile over the regimes; an order a
    regime lacks has no line there."""
    cfg = config or GateConfig()
    regimes = regimes_of(fits)
    n_fft, sr = pool.n_fft, pool.sr
    payloads = {
        name: _payload(fit, n_fft=n_fft, hop=pool.hop, sr=sr) for name, fit in regimes.items()
    }
    shapes = {name: line_shapes(pl, step_hz=cfg.shape_step_hz) for name, pl in payloads.items()}
    k_all = max(pl.k_prof for pl in payloads.values())
    df = sr / n_fft
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for w, st in zip(pool.windows, pool.starts, strict=True):
        n_rot, n = w.shape[0], st.size
        k = np.arange(1, k_all + 1)
        centres = np.clip(st + n_fft // 2, 0, w.shape[1] - 1)
        f = w[:, centres]  # (R, n) frame-centre speeds
        centre_hz = f[:, :, None] * k[None, None, :]  # (R, n, K) targets
        if n == 0:
            out.append((np.zeros(centre_hz.shape), centre_hz))
            continue
        bins = np.clip(np.rint(centre_hz / df).astype(np.int64), 0, n_fft // 2)
        t_bin = bins.transpose(1, 0, 2).reshape(n, -1)  # (n, T), T = R * K
        t_nu = t_bin * df
        t_r, t_k = np.divmod(np.arange(n_rot * k_all), k_all)
        line = np.zeros((n, n_rot * k_all))
        rest = np.zeros((n, n_rot * k_all))
        for name, wt in _regime_weights(regimes, w, centres, sr).items():
            if not bool(np.any(wt > 0.0)):
                continue
            pl, sh = payloads[name], shapes[name]
            kr = _k_full(regimes[name], w, sr)  # the model's orders on this window
            obs = pl.line_obs.numpy()  # (R, F)
            with torch.no_grad():
                floor = _floor_obs(pl, w, st).numpy()  # (n, F)
            flat = sh.shape[:, :kr].reshape(n_rot * kr, -1)
            q = np.arange(n_rot * kr)[None, :]  # every source line (s, j), flattened
            q_rot = q // kr
            s_centre = (f[:, :, None] * np.arange(1, kr + 1)).transpose(1, 0, 2).reshape(n, -1)
            s_amp = (sh.level[:, None, :kr] * sh.speed_law(f)[:, :, None]).transpose(1, 0, 2)
            s_amp = s_amp.reshape(n, -1)  # (n, Q)
            # a target (r, k) is its own source line q = r * kr + k - 1 where k <= kr
            has = t_k < kr
            t_q = np.where(has, t_r * kr + t_k, 0)
            rows = np.arange(n_rot * k_all)
            for i in range(n):
                h = _shape_at(flat, q, t_nu[i][:, None] - s_centre[i][None, :], sh.step_hz)
                h += _shape_at(flat, q, t_nu[i][:, None] + s_centre[i][None, :], sh.step_hz)
                c = s_amp[i][None, :] * obs[q_rot, t_bin[i][:, None]] * h  # (T, Q)
                own = np.where(has, c[rows, t_q], 0.0)
                line[i] += wt[i] * own
                rest[i] += wt[i] * (floor[i, t_bin[i]] + c.sum(axis=1) - own)
        tiny = np.finfo(np.float64).tiny
        contrast = 10.0 * np.log10((line + np.maximum(rest, tiny)) / np.maximum(rest, tiny))
        out.append((contrast.reshape(n, n_rot, k_all).transpose(1, 0, 2), centre_hz))
    return out


# -- the exact reference ------------------------------------------------------------


@dataclass
class Decomposition:
    """The mic-mean expected periodogram of frames and each line's part of it.

    ``power`` ``(n, F)``: ``S``, as observed (gains, transfer, array response);
    ``lines`` ``(R, n, K, F)``: every line's contribution to ``S``;
    ``floor`` ``(n, F)``: ``S`` minus every line (floor and wind);
    ``centre_rps`` ``(R, n)``: each rotor's speed at the frame centre;
    ``k_max``: orders computed.
    """

    power: np.ndarray
    lines: np.ndarray
    floor: np.ndarray
    centre_rps: np.ndarray
    freqs_hz: np.ndarray
    k_max: int


def _k_full(fit: dict[str, Any], rps: np.ndarray, sr: int) -> int:
    """Every in-band order of ``fit``'s profile on the rotors that spin in ``rps``."""
    k_prof = int(np.asarray(fit["params"]["profile"]["profile_db"]).shape[1])
    peak = rps.max(axis=1)
    peak = peak[peak > 0.0]
    if peak.size == 0:
        return 1
    return int(min(k_prof, SP.k_max_for_carrier(peak, sr, k_cap=k_prof)))


def decompose(
    fits: Mapping[str, dict[str, Any]] | dict[str, Any],
    rps: np.ndarray,
    starts: np.ndarray,
    *,
    k_max: int | None = None,
    n_fft: int = SP.FLIGHT_N_FFT,
    hop: int = SP.FLIGHT_HOP,
    sr: int = SP.FLIGHT_SR,
) -> Decomposition:
    """EXACT :class:`Decomposition` of ``fits`` on the frames at ``starts`` of
    ``rps`` ``(R, T)`` rev/s (:func:`spectrum.flight_cache`, chirp included):
    each regime's orders up to ``k_max``, or every in-band order of its own
    profile when ``None``. Mic mean of :func:`render.expected_periodogram`."""
    regimes = regimes_of(fits)
    rps = np.atleast_2d(np.asarray(rps, dtype=np.float64))
    starts = np.asarray(starts, dtype=np.int64)
    k_of = {
        name: _k_full(fit, rps, sr) if k_max is None else min(int(k_max), _k_full(fit, rps, sr))
        for name, fit in regimes.items()
    }
    kk = max(k_of.values())
    centres = np.clip(starts + int(n_fft) // 2, 0, rps.shape[1] - 1)
    weights = _regime_weights(regimes, rps, centres, sr)
    n_rotors, f_bins = rps.shape[0], int(n_fft) // 2 + 1
    power = np.zeros((starts.size, f_bins))
    lines = np.zeros((n_rotors, starts.size, kk, f_bins))
    with torch.no_grad():
        for regime, weight in weights.items():
            if not bool(np.any(weight > 0.0)):
                continue
            pl = _payload(regimes[regime], n_fft=n_fft, hop=hop, sr=sr)
            rate = SP.flight_rate_work(pl.grid, rps, starts)
            cache = SP.flight_cache(pl.grid, pl.params, rate_work=rate, k_max=k_of[regime])
            per_mic = SP.flight_model_cached(pl.grid, cache)  # (M, n, F), transfer + gains
            s = (per_mic * pl.array[:, None, :]).mean(dim=0)
            per_line = cache.lines * pl.line_obs[:, None, None, :]
            wt = torch.as_tensor(weight, dtype=torch.float64)
            power += (s * wt[:, None]).cpu().numpy()
            lines[:, :, : k_of[regime]] += (per_line * wt[None, :, None, None]).cpu().numpy()
    return Decomposition(
        power=power,
        lines=lines,
        floor=power - lines.sum(axis=(0, 2)),
        centre_rps=rps[:, centres],
        freqs_hz=np.fft.rfftfreq(int(n_fft), 1.0 / sr),
        k_max=kk,
    )


# -- coverage and the verdict ----------------------------------------------------------


def _eligibility(
    rps: np.ndarray, starts: np.ndarray, n_fft: int, cfg: GateConfig
) -> tuple[np.ndarray, ...]:
    """``(eligible, stopped, close)``, each ``(R, n)`` bool, over the whole frame."""
    idx = starts[:, None] + np.arange(int(n_fft))[None, :]  # (n, n_fft)
    seg = rps[:, idx]  # (R, n, n_fft)
    stopped = seg.min(axis=2) <= cfg.spin_min_rps
    n_rotors = rps.shape[0]
    close = np.zeros_like(stopped)
    for r in range(n_rotors):
        for s in range(n_rotors):
            if s != r:
                close[r] |= np.abs(seg[r] - seg[s]).min(axis=1) <= cfg.sep_rps
    return ~stopped & ~close, stopped, close & ~stopped


@dataclass(frozen=True)
class Coverage:
    """Which frames of a pool can test each rotor, under a config.

    ``masks[i]`` ``(R, n_i)``: eligible frames of window ``i``; ``eligible``,
    ``stopped``, ``close`` ``(R,)``: counts over the pool; ``untestable``: why
    the pool cannot test some rotor, else ``None``. A property of the POOL.
    """

    masks: tuple[np.ndarray, ...]
    eligible: np.ndarray
    stopped: np.ndarray
    close: np.ndarray
    untestable: str | None


def pool_coverage(pool: FramePool, config: GateConfig | None = None) -> Coverage:
    """:class:`Coverage` of ``pool``: stopped and near-coincident frames per
    rotor, and whether every rotor keeps ``min_eligible_frames``."""
    cfg = config or GateConfig()
    per = [
        _eligibility(w, st, pool.n_fft, cfg)
        for w, st in zip(pool.windows, pool.starts, strict=True)
    ]
    eligible = np.concatenate([p[0] for p in per], axis=1).sum(axis=1)
    short = np.flatnonzero(eligible < int(cfg.min_eligible_frames))
    why = None
    if short.size:
        r = int(short[0])
        why = (
            f"untestable: rotor {r + 1} eligible in {eligible[r]}/{pool.n_frames} frames "
            f"(needs {cfg.min_eligible_frames})"
        )
    return Coverage(
        masks=tuple(p[0] for p in per),
        eligible=eligible,
        stopped=np.concatenate([p[1] for p in per], axis=1).sum(axis=1),
        close=np.concatenate([p[2] for p in per], axis=1).sum(axis=1),
        untestable=why,
    )


@dataclass
class GateResult:
    """The gate's verdict and the counts behind it (per rotor, over the pool)."""

    passed: bool
    reason: str
    n_frames: int
    eligible: np.ndarray
    stopped: np.ndarray
    close: np.ndarray
    frames_ok: np.ndarray
    config: GateConfig = field(default_factory=GateConfig)

    @property
    def pass_frac(self) -> np.ndarray:
        return self.frames_ok / np.maximum(self.eligible, 1)

    def summary(self) -> dict[str, Any]:
        return dict(
            passed=self.passed,
            reason=self.reason,
            n_frames=self.n_frames,
            eligible=self.eligible.tolist(),
            excused_stopped=self.stopped.tolist(),
            excused_close=self.close.tolist(),
            frames_ok=self.frames_ok.tolist(),
            pass_frac=np.round(self.pass_frac, 4).tolist(),
            config=asdict(self.config),
        )


def gate(
    fits: Mapping[str, dict[str, Any]] | dict[str, Any],
    pool: FramePool,
    config: GateConfig | None = None,
) -> GateResult:
    """Judge ``fits`` (one payload or a ``standby``/``cruise`` pair) on ``pool``."""
    cfg = config or GateConfig()
    cov = pool_coverage(pool, cfg)
    n_rotors = pool.n_rotors
    zeros = np.zeros(n_rotors, dtype=np.int64)

    def result(passed: bool, reason: str, ok: np.ndarray) -> GateResult:
        return GateResult(
            passed=passed,
            reason=reason,
            n_frames=pool.n_frames,
            eligible=cov.eligible,
            stopped=cov.stopped,
            close=cov.close,
            frames_ok=ok,
            config=cfg,
        )

    if cov.untestable is not None:
        return result(False, cov.untestable, zeros)
    ok = zeros.copy()
    for (contrast, centre_hz), e in zip(line_contrasts(fits, pool, cfg), cov.masks, strict=True):
        good = (
            (contrast >= cfg.threshold_db)
            & (centre_hz >= cfg.f_min_hz)
            & (centre_hz <= cfg.f_max_hz)
        )
        passed = good.sum(axis=2) >= cfg.min_orders  # (R, n)
        ok += (e & passed).sum(axis=1)
    need = np.ceil(cfg.pass_frac * cov.eligible - 1e-9).astype(np.int64)
    short = np.flatnonzero(ok < need)
    if short.size:
        r = int(short[0])
        why = f"rotor {r + 1} passed {ok[r]} of {cov.eligible[r]} eligible frames (needs {need[r]})"
        return result(False, why, ok)
    return result(True, "every rotor identifiable", ok)
