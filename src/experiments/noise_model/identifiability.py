"""Per-rotor IDENTIFIABILITY gate of a v3 rig payload on a trajectory pool.

The rule (agreed with Dmitrii, 2026-09-29; ``docs/experiments/noise-model-v3.md``
§ "Identifiability gate"):

* A harmonic ``k`` of rotor ``r`` is IDENTIFIABLE in a frame when removing its
  line changes the frame's expected spectrum at the line centre by at least
  ``threshold_db``: ``10 log10(S / (S - L_rk)) >= T`` at the bin nearest
  ``k f_r``, ``S`` the mic-mean expected periodogram (floor, wind, every line of
  every rotor), ``L_rk`` that one line's contribution to it. Any order in
  ``[f_min_hz, f_max_hz]`` counts; a lone line (DREGON's order 70) counts.
* A rotor PASSES a frame when at least ``min_orders`` of its harmonics are
  identifiable there. Frames are the STFT frames the models read
  (``n_fft`` 2048, ``hop`` 512 at 16 kHz); the rotor speed moves inside the
  frame exactly as the renderer's kernel moves it (chirped atoms).
* A rotor is EXCUSED in a frame that no rig can pass: stopped
  (``spin_min_rps``) or closer than ``sep_rps`` to another rotor anywhere in
  the frame. The distance is geometry only, never the rig's line width.
* A rig PASSES when every rotor passes at least ``pass_frac`` of its eligible
  frames AND has at least ``min_eligible_frames`` of them: a pool that rarely
  separates a rotor makes it UNTESTABLE (:func:`pool_coverage`, a property of
  the pool), which fails - the gate is never passed vacuously.

The spectra are the forward model's own, not a restatement:
:func:`spectrum.flight_cache` (every line's expected periodogram apart, the
floor, the wind, the mic gains) and :func:`spectrum.flight_model_cached` (their
sum, which :func:`spectrum.flight_model` equals to rounding), per regime,
power-blended per frame by :func:`regime_blend_weight` exactly as
:func:`render.expected_periodogram_regimes` does. The array response is
absent: it averages out of the mic mean.

Two exact economies. (1) Orders are evaluated PROGRESSIVELY: pass 1 builds the
lines up to ``first_k`` and counts only the lines whose background is complete
(centre below the lowest uncomputed line minus ``margin_hz``); pass 2 recounts
over every in-band order on the frames where an eligible rotor is still short.
(2) The pool is a fixed-size frame subsample (:func:`frame_pool`), and the
evaluation stops as soon as some rotor's failures exceed its budget.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from data_processing.noise_model.params import check_schema
from data_processing.noise_model.render import fit_work_rate, regime_blend_weight

from . import model as MD
from . import spectrum as SP

__all__ = [
    "Coverage",
    "Decomposition",
    "FramePool",
    "GateConfig",
    "GateResult",
    "decompose",
    "frame_pool",
    "gate",
    "pool_coverage",
    "regimes_of",
]


@dataclass(frozen=True)
class GateConfig:
    """Every number the gate reads. Defaults: the agreed rule."""

    #: ``T``: dB the line must add to the spectrum at its centre bin.
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
    #: Pass 1's order count; pass 2 takes every in-band order.
    first_k: int = 24
    #: Pass 1 counts a line only this far below the lowest uncomputed line.
    margin_hz: float = 200.0
    n_fft: int = SP.FLIGHT_N_FFT
    hop: int = SP.FLIGHT_HOP
    sr: int = SP.FLIGHT_SR
    frame_chunk: int = 16


@dataclass(frozen=True)
class FramePool:
    """Frames drawn from trajectory windows: ``windows[i]`` is ``(R, T)`` rev/s at
    ``sr``, ``starts[i]`` the sample starts of its frames kept in the pool."""

    windows: tuple[np.ndarray, ...]
    starts: tuple[np.ndarray, ...]
    seed: int

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
) -> FramePool:
    """``n_frames`` frames drawn uniformly without replacement from every
    STFT frame (``data.periodogram``'s framing) of ``windows``."""
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
    return FramePool(windows=wins, starts=starts, seed=int(seed))


def regimes_of(fits: Mapping[str, dict[str, Any]] | dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``{"single": fit}`` for one payload, or the ``standby``/``cruise`` pair."""
    if "params" in fits:
        return {"single": dict(fits)}
    keys = set(fits)
    if keys == {"single"} or keys == {"standby", "cruise"}:
        return {str(k): dict(v) for k, v in fits.items()}
    raise ValueError(f"expected a payload, 'single', or 'standby'+'cruise'; got {sorted(keys)}")


@dataclass
class Decomposition:
    """The mic-mean expected periodogram of frames and each line's part of it.

    ``power`` ``(n, F)``: ``S``, as observed (transfer and gains applied);
    ``lines`` ``(R, n, K, F)``: every line's contribution to ``S``;
    ``floor`` ``(n, F)``: ``S`` minus every computed line (floor and wind);
    ``centre_rps`` ``(R, n)``: each rotor's speed at the frame centre;
    ``k_max``: orders computed (``S`` holds only these lines).
    """

    power: np.ndarray
    lines: np.ndarray
    floor: np.ndarray
    centre_rps: np.ndarray
    freqs_hz: np.ndarray
    k_max: int


def _k_full(fit: dict[str, Any], rps: np.ndarray, sr: int) -> int:
    """Every in-band order of ``fit``'s profile, on the rotors that spin in
    ``rps`` (a rotor that never spins has no line; a window where none spins, one)."""
    k_prof = int(np.asarray(fit["params"]["profile"]["profile_db"]).shape[1])
    peak = rps.max(axis=1)
    peak = peak[peak > 0.0]
    if peak.size == 0:
        return 1
    return int(min(k_prof, SP.k_max_for_carrier(peak, sr, k_cap=k_prof)))


def _k_all(regimes: Mapping[str, dict[str, Any]], rps: np.ndarray, sr: int) -> int:
    """The largest :func:`_k_full` over the regimes: all lines of every regime."""
    return max(_k_full(f, rps, sr) for f in regimes.values())


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
    """:class:`Decomposition` of ``fits`` on the frames at ``starts`` of ``rps``
    ``(R, T)`` rev/s: each regime's orders up to ``k_max``, or every in-band
    order of its own profile when ``None``."""
    regimes = regimes_of(fits)
    rps = np.atleast_2d(np.asarray(rps, dtype=np.float64))
    starts = np.asarray(starts, dtype=np.int64)
    k_of = {
        name: _k_full(fit, rps, sr) if k_max is None else min(int(k_max), _k_full(fit, rps, sr))
        for name, fit in regimes.items()
    }
    kk = max(k_of.values())
    centres = np.clip(starts + int(n_fft) // 2, 0, rps.shape[1] - 1)
    if "single" in regimes:
        weights = {"single": np.ones(starts.size)}
    else:
        w = regime_blend_weight(rps, sr=sr)[centres]
        weights = {"standby": 1.0 - w, "cruise": w}
    n_rotors, f_bins = rps.shape[0], int(n_fft) // 2 + 1
    power = np.zeros((starts.size, f_bins))
    lines = np.zeros((n_rotors, starts.size, kk, f_bins))
    with torch.no_grad():
        for regime, weight in weights.items():
            if not bool(np.any(weight > 0.0)):
                continue
            fit = regimes[regime]
            p = check_schema(fit)
            grid = SP.flight_grid(sr=sr, n_fft=n_fft, hop=hop, sr_work=fit_work_rate(fit))
            params = MD.params_from_dict(p)
            k_reg = k_of[regime]
            rate = SP.flight_rate_work(grid, rps, starts)
            cache = SP.flight_cache(grid, params, rate_work=rate, k_max=k_reg)
            s = SP.flight_model_cached(grid, cache).mean(dim=0)  # (n, F), mic mean
            # each rotor's line gain as observed: mic mean of all_gain_m * line_gain_mr
            g = (cache.all_gain[:, None] * cache.line_gain).mean(dim=0)  # (R,)
            per_line = cache.lines * (g[:, None, None, None] * grid.transfer_power)
            wt = torch.as_tensor(weight, dtype=torch.float64)
            power += (s * wt[:, None]).cpu().numpy()
            lines[:, :, :k_reg] += (per_line * wt[None, :, None, None]).cpu().numpy()
    return Decomposition(
        power=power,
        lines=lines,
        floor=power - lines.sum(axis=(0, 2)),
        centre_rps=rps[:, centres],
        freqs_hz=np.fft.rfftfreq(int(n_fft), 1.0 / sr),
        k_max=kk,
    )


def _identifiable_counts(dec: Decomposition, cfg: GateConfig, *, complete_all: bool) -> np.ndarray:
    """``(R, n)`` identifiable harmonics per rotor and frame of ``dec``."""
    n_rotors, n, kk, _f = dec.lines.shape
    k = np.arange(1, kk + 1, dtype=np.float64)
    centre_hz = dec.centre_rps[:, :, None] * k[None, None, :]  # (R, n, K)
    bins = np.rint(centre_hz * cfg.n_fft / cfg.sr).astype(np.int64)
    bins = np.clip(bins, 0, dec.power.shape[1] - 1)
    ri, ni = np.meshgrid(np.arange(n_rotors), np.arange(n), indexing="ij")
    line = dec.lines[ri[..., None], ni[..., None], np.arange(kk)[None, None, :], bins]
    s = dec.power[ni[..., None], bins]
    rest = np.maximum(s - line, np.finfo(np.float64).tiny)
    contrast = 10.0 * np.log10(np.maximum(s, np.finfo(np.float64).tiny) / rest)
    ok = (contrast >= cfg.threshold_db) & (centre_hz >= cfg.f_min_hz) & (centre_hz <= cfg.f_max_hz)
    if not complete_all:
        # the lowest line NOT computed in this frame: order kk + 1 of the slowest rotor
        spinning = np.where(dec.centre_rps > 0.0, dec.centre_rps, np.inf).min(axis=0)
        edge = (kk + 1) * spinning - cfg.margin_hz  # (n,); inf where nothing spins
        ok &= centre_hz <= edge[None, :, None]
    return ok.sum(axis=2)


def _eligibility(rps: np.ndarray, starts: np.ndarray, cfg: GateConfig) -> tuple[np.ndarray, ...]:
    """``(eligible, stopped, close)``, each ``(R, n)`` bool, over the whole frame."""
    idx = starts[:, None] + np.arange(int(cfg.n_fft))[None, :]  # (n, n_fft)
    seg = rps[:, idx]  # (R, n, n_fft)
    stopped = seg.min(axis=2) <= cfg.spin_min_rps
    n_rotors = rps.shape[0]
    close = np.zeros_like(stopped)
    for r in range(n_rotors):
        for s in range(n_rotors):
            if s != r:
                close[r] |= np.abs(seg[r] - seg[s]).min(axis=1) <= cfg.sep_rps
    return ~stopped & ~close, stopped, close & ~stopped


@dataclass
class GateResult:
    """The gate's verdict and the counts behind it (per rotor, over the pool)."""

    passed: bool
    reason: str
    n_frames: int
    eligible: np.ndarray
    stopped: np.ndarray
    close: np.ndarray
    evaluated: np.ndarray
    frames_ok: np.ndarray
    aborted: bool
    config: GateConfig = field(default_factory=GateConfig)

    @property
    def coverage(self) -> np.ndarray:
        return self.eligible / max(self.n_frames, 1)

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
            coverage=np.round(self.coverage, 4).tolist(),
            aborted=self.aborted,
            config=asdict(self.config),
        )


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
    per = [_eligibility(w, st, cfg) for w, st in zip(pool.windows, pool.starts, strict=True)]
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


def gate(
    fits: Mapping[str, dict[str, Any]] | dict[str, Any],
    pool: FramePool,
    config: GateConfig | None = None,
) -> GateResult:
    """Judge ``fits`` (one payload or a ``standby``/``cruise`` pair) on ``pool``."""
    cfg = config or GateConfig()
    regimes = regimes_of(fits)
    n_rotors, n_total = pool.n_rotors, pool.n_frames
    cov = pool_coverage(pool, cfg)
    elig, eligible = cov.masks, cov.eligible
    zeros = np.zeros(n_rotors, dtype=np.int64)

    def result(passed: bool, reason: str, evaluated, ok, aborted: bool) -> GateResult:
        return GateResult(
            passed=passed,
            reason=reason,
            n_frames=n_total,
            eligible=eligible,
            stopped=cov.stopped,
            close=cov.close,
            evaluated=evaluated,
            frames_ok=ok,
            aborted=aborted,
            config=cfg,
        )

    if cov.untestable is not None:
        return result(False, cov.untestable, zeros, zeros, False)

    budget = np.floor((1.0 - cfg.pass_frac) * eligible + 1e-9).astype(np.int64)
    ok = np.zeros(n_rotors, dtype=np.int64)
    fail = np.zeros(n_rotors, dtype=np.int64)
    evaluated = np.zeros(n_rotors, dtype=np.int64)
    for w, st, e in zip(pool.windows, pool.starts, elig, strict=True):
        k_full = _k_all(regimes, w, cfg.sr)
        for c0 in range(0, st.size, int(cfg.frame_chunk)):
            sl = slice(c0, c0 + int(cfg.frame_chunk))
            s_chunk, e_chunk = st[sl], e[:, sl]
            if not e_chunk.any():
                continue
            passed = np.zeros_like(e_chunk)
            k1 = min(int(cfg.first_k), k_full)
            dec = decompose(regimes, w, s_chunk, k_max=k1, n_fft=cfg.n_fft, hop=cfg.hop, sr=cfg.sr)
            passed |= _identifiable_counts(dec, cfg, complete_all=k1 >= k_full) >= cfg.min_orders
            todo = (e_chunk & ~passed).any(axis=0)
            if k1 < k_full and todo.any():
                dec = decompose(
                    regimes, w, s_chunk[todo], k_max=k_full, n_fft=cfg.n_fft, hop=cfg.hop, sr=cfg.sr
                )
                counts = _identifiable_counts(dec, cfg, complete_all=True)
                passed[:, todo] |= counts >= cfg.min_orders
            ok += (e_chunk & passed).sum(axis=1)
            fail += (e_chunk & ~passed).sum(axis=1)
            evaluated += e_chunk.sum(axis=1)
            over = np.flatnonzero(fail > budget)
            if over.size:
                r = int(over[0])
                why = (
                    f"rotor {r + 1} failed {fail[r]} frames, budget {budget[r]} "
                    f"of {eligible[r]} eligible"
                )
                return result(False, why, evaluated, ok, True)
    return result(True, "every rotor identifiable", evaluated, ok, False)
