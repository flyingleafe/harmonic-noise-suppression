"""The FROZEN rotor-speed statistics, discrepancy and acceptance rule.

This module is the campaign's judge. Every candidate trajectory model — from
the first i.i.d. strawman to whatever ends up in the paper — is summarised by
the same :class:`TrajStats`, compared to the real per-rig statistics by the
same five-family :func:`discrepancy`, and accepted or rejected by the same
:func:`passes`. Changing any definition here invalidates every number the
campaign has produced, so the definitions are spelled out in full:

**Support.** Only samples inside
:func:`~experiments.rps_traj.data.airborne_segments` of a flight enter any
statistic, on the 100 Hz common grid
(:data:`~experiments.rps_traj.data.RATE_HZ`). Synthetic trajectories go
through the identical segmentation, so a model that never leaves the ground
is scored on nothing and a model with unrealistic transients is scored on the
same eroded cruise the real rigs are scored on.

**Families.**

1. ``overall_mean`` — mean rev/s over all airborne samples of all rotors,
   sample-weighted (i.e. duration-weighted).
2. ``rotor_mean`` — the same per rotor.
3. ``rotor_var`` — per-rotor variance over ALL airborne samples POOLED over
   flights, about the pooled per-rotor mean. Pooling on purpose: a model that
   reproduces within-flight wobble but puts every flight at the same hover
   level is wrong, and this is the family that says so.
4. ``acf`` — per segment, per rotor, within-segment demeaned, the *biased*
   normalised autocorrelation
   ``sum_t x_t x_{t+k} / sum_t x_t^2`` at :data:`LAGS_S`, averaged over
   segments with weight = segment length. A segment shorter than twice a lag
   contributes nothing at that lag (its biased estimate would be more taper
   than signal).
5. ``xcorr`` — zero-lag Pearson correlation between rotors on
   within-segment-demeaned data pooled over all segments.

NaNs are dropped pairwise everywhere (a missing telemetry sample removes the
products it takes part in, nothing else).

**What the short ACF lags really measure.** The telemetry behind both rigs is
trustworthy only below ~5 Hz (``results/rps_traj/explore/summary.json``), and
the seven :data:`LAGS_S` entries below 0.2 s are biased by the loggers in
OPPOSITE directions:

- michaels is INFLATED. Its DatCon log is a 29.41 Hz sampling of an
  asynchronous ~17 Hz ESC update with no anti-alias roll-off, so a 0.05 s lag
  spans 1.47 native samples: on the 100 Hz grid that lag is mostly this
  module's own linear interpolation, not data.
- DREGON is DEFLATED. ``motor.measured`` updates ~45 times/s in 0.30 rev/s
  steps (95 % of consecutive 1 kHz samples are bit-identical, median hold
  19 ms); the quantise-and-hold residual is near-white (~3e-2 (rev/s)^2/Hz
  over 0.5-10 Hz) and additive noise of variance v scales every nonzero-lag
  ACF by 1 / (1 + v / sigma^2).

The size of it, from DREGON room1's within-flight measured-vs-command A/B:
rotor-mean ACF 0.907 vs 0.978 at 0.02 s and 0.78 vs 0.87 at 0.1 s, converged
by ~0.2 s — a ~0.07-0.09 short-lag bias that vanishes exactly where the
logger-limited lags end. Those lags stay in :data:`LAGS_S` because a model
must reproduce what the data does, but a candidate that matches lags
>= 0.2 s and misses only the shortest ones is closer to the truth than its
``acf`` number alone suggests.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.rps_traj.data import RATE_HZ, Flight, airborne_segments

#: ACF lags in samples of the 100 Hz grid: 24 log-spaced lags from 0.05 s to
#: 10 s, rounded to the grid. :data:`LAGS_S` is the realised (grid-exact) lag
#: in seconds — the rounding is part of the frozen definition.
LAG_SAMPLES = np.maximum(1, np.round(np.geomspace(0.05, 10.0, 24) * RATE_HZ).astype(np.int64))
LAGS_S = LAG_SAMPLES / RATE_HZ

#: The five discrepancy families, in report order. Frozen.
FAMILIES = ("overall_mean", "rotor_mean", "rotor_var", "acf", "xcorr")

#: An improvement counts as strict only beyond this, and a regression only
#: beyond this is a regression.
STRICT_EPS = 1e-9

#: :func:`passes` needs a strict improvement in at least this many families.
MIN_STRICT_FAMILIES = 3


@dataclass
class TrajStats:
    """The frozen summary of a set of trajectories (quad: 4 rotors, 24 lags)."""

    overall_mean: float
    rotor_mean: np.ndarray  # (n_rotors,) rev/s
    rotor_var: np.ndarray  # (n_rotors,) (rev/s)^2
    acf: np.ndarray  # (n_rotors, len(LAGS_S))
    xcorr: np.ndarray  # (n_rotors, n_rotors)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict; non-finite entries become ``null``."""
        return {
            "overall_mean": _jsonable(self.overall_mean),
            "rotor_mean": _jsonable(self.rotor_mean),
            "rotor_var": _jsonable(self.rotor_var),
            "acf": _jsonable(self.acf),
            "xcorr": _jsonable(self.xcorr),
            "lags_s": LAGS_S.tolist(),
        }

    def to_json(self) -> str:
        """JSON text (NaN is written as ``null``, so it round-trips)."""
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrajStats:
        lags = payload.get("lags_s")
        if lags is not None and not np.allclose(
            np.asarray(lags, dtype=np.float64), LAGS_S, rtol=0, atol=1e-12
        ):
            raise ValueError(
                "stats were computed on different ACF lags than the frozen LAGS_S "
                "— they are not comparable"
            )
        return cls(
            overall_mean=_as_float(payload["overall_mean"]),
            rotor_mean=_as_array(payload["rotor_mean"]),
            rotor_var=_as_array(payload["rotor_var"]),
            acf=_as_array(payload["acf"]),
            xcorr=_as_array(payload["xcorr"]),
        )

    @classmethod
    def from_json(cls, text: str | bytes) -> TrajStats:
        return cls.from_dict(json.loads(text))

    @property
    def n_rotors(self) -> int:
        return int(np.asarray(self.rotor_mean).size)


def _jsonable(value: Any) -> Any:
    """float / array → JSON value, with ``None`` in place of NaN and inf."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr) if np.isfinite(arr) else None
    return [_jsonable(v) for v in arr]


def _as_float(value: Any) -> float:
    return float("nan") if value is None else float(value)


def _as_array(value: Any) -> np.ndarray:
    """Nested lists with ``null`` for NaN → float64 array (``None`` → NaN)."""
    return np.array(value, dtype=np.float64)


# ─── compute_stats ────────────────────────────────────────────────────────────


def _airborne_arrays(flights: Sequence[Flight]) -> Iterator[np.ndarray]:
    """Every airborne segment of every flight as a ``(n_rotors, n)`` array."""
    for flight in flights:
        rps = np.atleast_2d(np.asarray(flight.rps, dtype=np.float64))
        for sl in airborne_segments(rps, flight.fs):
            yield rps[:, sl]


def compute_stats(flights: Sequence[Flight]) -> TrajStats:
    """The frozen statistics of a set of flights (see the module docstring)."""
    segments = list(_airborne_arrays(flights))
    if not segments:
        raise ValueError("no airborne segments in the given flights — nothing to summarise")
    n_rotors = segments[0].shape[0]
    if any(seg.shape[0] != n_rotors for seg in segments):
        raise ValueError("flights disagree on the rotor count")
    n_lags = LAG_SAMPLES.size

    # Pooled first and second moments per rotor (means and variance).
    s1 = np.zeros(n_rotors)
    s2 = np.zeros(n_rotors)
    cnt = np.zeros(n_rotors)
    # Duration-weighted ACF accumulators.
    acf_num = np.zeros((n_rotors, n_lags))
    acf_w = np.zeros((n_rotors, n_lags))
    # Pairwise cross-correlation accumulators on within-segment-demeaned data:
    # sums over the samples where BOTH rotors of the pair are finite.
    pair_xy = np.zeros((n_rotors, n_rotors))
    pair_xx = np.zeros((n_rotors, n_rotors))  # [i, j] = sum of rotor i's square

    for seg in segments:
        finite = np.isfinite(seg)
        vals = np.where(finite, seg, 0.0)
        s1 += vals.sum(axis=1)
        s2 += (vals * vals).sum(axis=1)
        cnt += finite.sum(axis=1)

        n = seg.shape[1]
        seg_mean = np.where(
            finite.any(axis=1), vals.sum(axis=1) / np.maximum(finite.sum(axis=1), 1), np.nan
        )
        centred = seg - seg_mean[:, None]
        cfin = np.isfinite(centred)
        c = np.where(cfin, centred, 0.0)

        c0 = (c * c).sum(axis=1)
        ok = c0 > 0.0
        for j, k in enumerate(LAG_SAMPLES):
            lag = int(k)
            if n < 2 * lag:
                continue
            num = (c[:, : n - lag] * c[:, lag:]).sum(axis=1)
            acf_num[ok, j] += float(n) * num[ok] / c0[ok]
            acf_w[ok, j] += float(n)

        for i in range(n_rotors):
            pair_xy[i, i] += c0[i]
            pair_xx[i, i] += c0[i]
            for j in range(i + 1, n_rotors):
                both = cfin[i] & cfin[j]
                xi = centred[i, both]
                xj = centred[j, both]
                pair_xy[i, j] += float(xi @ xj)
                pair_xx[i, j] += float(xi @ xi)
                pair_xx[j, i] += float(xj @ xj)

    with np.errstate(invalid="ignore", divide="ignore"):
        rotor_mean = np.where(cnt > 0, s1 / np.maximum(cnt, 1), np.nan)
        rotor_var = np.where(cnt > 0, s2 / np.maximum(cnt, 1) - rotor_mean**2, np.nan)
        rotor_var = np.maximum(rotor_var, 0.0)
        total = cnt.sum()
        overall_mean = float(s1.sum() / total) if total > 0 else float("nan")
        acf = np.where(acf_w > 0, acf_num / np.maximum(acf_w, 1.0), np.nan)

    xcorr = np.full((n_rotors, n_rotors), np.nan)
    for i in range(n_rotors):
        for j in range(i, n_rotors):
            denom = pair_xx[i, j] * pair_xx[j, i]
            value = pair_xy[i, j] / np.sqrt(denom) if denom > 0 else np.nan
            xcorr[i, j] = xcorr[j, i] = value

    return TrajStats(
        overall_mean=overall_mean,
        rotor_mean=rotor_mean,
        rotor_var=rotor_var,
        acf=acf,
        xcorr=xcorr,
    )


# ─── discrepancy and acceptance ───────────────────────────────────────────────


def _rms(diff: np.ndarray) -> float:
    """RMS over the finite entries (pairwise NaN drop); NaN if none are."""
    d = np.asarray(diff, dtype=np.float64).reshape(-1)
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d * d))) if d.size else float("nan")


def _log_ratio(model: np.ndarray, real: np.ndarray) -> np.ndarray:
    """``|ln(model/real)|``, ``inf`` where either variance is non-positive."""
    model = np.asarray(model, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    out = np.full(model.shape, np.inf)
    ok = (model > 0) & (real > 0)
    out[ok] = np.abs(np.log(model[ok] / real[ok]))
    nan = ~np.isfinite(model) | ~np.isfinite(real)
    out[nan] = np.nan
    return out


def discrepancy(model: TrajStats, real: TrajStats) -> dict[str, float]:
    """The five-family distance from ``model`` statistics to ``real`` ones.

    - ``overall_mean``: ``|Δ|`` in rev/s;
    - ``rotor_mean``: RMS over rotors of ``|Δμ_r|`` in rev/s;
    - ``rotor_var``: RMS over rotors of ``|ln(var_model / var_real)|``
      (scale-free, so a rig hovering at 90 rev/s and one at 300 rev/s are
      judged on the same footing);
    - ``acf``: RMS over rotors × lags of ``|Δρ|``;
    - ``xcorr``: RMS over the 6 off-diagonal rotor pairs of ``|Δc|``.

    Lower is better, zero is perfect. Entries missing on either side (a lag no
    segment was long enough for) are dropped from that family's RMS.
    """
    if model.n_rotors != real.n_rotors:
        raise ValueError(f"rotor count mismatch: {model.n_rotors} vs {real.n_rotors}")
    off = ~np.eye(real.n_rotors, dtype=bool)
    return {
        "overall_mean": abs(float(model.overall_mean) - float(real.overall_mean)),
        "rotor_mean": _rms(np.asarray(model.rotor_mean) - np.asarray(real.rotor_mean)),
        "rotor_var": _rms(_log_ratio(model.rotor_var, real.rotor_var)),
        "acf": _rms(np.asarray(model.acf) - np.asarray(real.acf)),
        "xcorr": _rms((np.asarray(model.xcorr) - np.asarray(real.xcorr))[off]),
    }


def passes(new: dict[str, float], base: dict[str, float]) -> tuple[bool, dict[str, bool]]:
    """Acceptance rule: no family may regress, and ≥3 must strictly improve.

    Returns ``(verdict, per_family_ok)`` where ``per_family_ok[f]`` is True
    when family ``f`` did not regress (``new <= base`` within
    :data:`STRICT_EPS`). A family improves *strictly* when it is smaller by
    more than :data:`STRICT_EPS`; the verdict needs
    :data:`MIN_STRICT_FAMILIES` = 3 of those on top of no regression
    anywhere, which is what stops a candidate from trading ACF for variance
    and calling it progress.
    """
    ok: dict[str, bool] = {}
    strict = 0
    for family in FAMILIES:
        n = float(new[family])
        b = float(base[family])
        ok[family] = n <= b + STRICT_EPS
        if n < b - STRICT_EPS:
            strict += 1
    verdict = all(ok.values()) and strict >= MIN_STRICT_FAMILIES
    return verdict, ok


# ─── model statistics from a sampler ──────────────────────────────────────────

#: A trajectory sampler: ``sampler(n_samples, rng) -> (n_rotors, n_samples)``
#: rev/s on the 100 Hz grid. All randomness MUST come from ``rng`` — that is
#: what makes :func:`stats_from_samples` reproducible.
Sampler = Callable[[int, np.random.Generator], np.ndarray]


def stats_from_samples(
    sampler: Sampler,
    durations_s: Sequence[float],
    fs: float = RATE_HZ,
    n_rep: int = 5,
    seed: int = 0,
) -> TrajStats:
    """Model statistics: draw the real flight durations ``n_rep`` times over.

    ``sampler(n_samples, rng)`` returns a ``(n_rotors, n_samples)`` rev/s
    trajectory and takes all its randomness from ``rng``; one call per
    duration in ``durations_s``, repeated ``n_rep`` times with independent
    generators derived from ``seed`` (``default_rng([seed, rep, i])``), so the
    result is a deterministic function of ``(sampler, durations_s, seed)``.

    Matching the real durations matters: the ACF is biased-normalised and
    segment-length weighted, so statistics of 20 s flights are not statistics
    of 200 s flights. The samples are run through
    :func:`~experiments.rps_traj.data.airborne_segments` exactly like real
    telemetry — a model that samples ground-level speeds loses those samples
    just as a real flight does.
    """
    flights: list[Flight] = []
    for rep in range(int(n_rep)):
        for i, duration in enumerate(durations_s):
            n = int(round(float(duration) * float(fs)))
            if n <= 0:
                continue
            rng = np.random.default_rng([int(seed), int(rep), int(i)])
            rps = np.asarray(sampler(n, rng), dtype=np.float64)
            if rps.ndim != 2 or rps.shape[1] != n:
                raise ValueError(f"sampler returned {rps.shape}, expected (n_rotors, {n}) rev/s")
            flights.append(
                Flight(
                    rig="model",
                    flight=f"rep{rep:02d}_flight{i:03d}",
                    fs=float(fs),
                    t0=0.0,
                    rps=rps,
                    source="sampler",
                )
            )
    return compute_stats(flights)


__all__ = [
    "FAMILIES",
    "LAGS_S",
    "LAG_SAMPLES",
    "MIN_STRICT_FAMILIES",
    "STRICT_EPS",
    "Sampler",
    "TrajStats",
    "compute_stats",
    "discrepancy",
    "passes",
    "stats_from_samples",
]
