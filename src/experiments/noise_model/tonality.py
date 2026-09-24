"""How TONAL is a v2 payload? Line-to-local-floor prominence, order by order.

The bank guards (``rig_sampler.check_sample``) constrain the trend's total
DROP, the 1/3-octave envelope, the line widths and finiteness. None of them
says a word about how far a rotor order stands above the broadband floor
AROUND it, which is what makes a noise "harmonic" to a model that must find
the comb. This module measures exactly that, on the noise-free EXPECTED
periodogram, so a whole 4096-entry bank can be audited without rendering audio.

The estimator is round 4's (``results/noise_v2/rounds/round4/legacy_truth/``
``anatomy.md`` § "MEASURED on the render: prominence over the local floor"),
applied to the expectation instead of a render:

* **peak** = the largest ``P_tot`` bin within +-1 bin of ``k f_r``;
* **local floor** = the MEDIAN of ``P_tot`` over the two-sided annulus at
  ``0.45 .. 0.7 fbar`` from ``k f_r`` (``fbar`` = the pattern's rotor-mean
  carrier), excluding +-1 bin around every rotor's ``k-1 / k / k+1`` lines;
* **prominence** = ``10 log10(peak / floor)`` in dB.

The floor it reads is therefore the AGGREGATE local floor: the broadband floor
plus whatever the other three rotors' skirts and their own lines contribute
between this rotor's orders. That is deliberate — it is what a separator sees —
but it means the number is NOT "line over broadband floor" for a rig whose
rotors are far apart in speed, and it is not directly comparable to a
single-rotor bench measurement.

Two carrier conventions matter and both are wrong in opposite directions:
four rotors at ONE speed pile every rotor's order on the same bin and overstate
visibility; one rotor alone understates the floor. This module therefore probes
with REAL four-rotor patterns — the per-rotor mean carriers of the campaign's
own frozen windows (:func:`select_patterns`).

Everything here works on ``(M, F)`` expected periodograms from
:class:`PatternProbe`, which is :class:`rig_sampler.ModelProbe` with a per-rotor
carrier row instead of one shared speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.noise_model import rig_sampler as RS

__all__ = [
    "ANNULUS_FRAC",
    "COMB_OFF_DB",
    "NAMED_ORDERS",
    "PROM_THRESHOLDS_DB",
    "Pattern",
    "PatternGeometry",
    "PatternProbe",
    "comb_pedestal_db",
    "comb_switch_check",
    "entry_row",
    "local_floor",
    "patterns_from_dicts",
    "payload_shape",
    "payload_stats",
    "profile_trend",
    "prominence_db",
    "real_profile_slopes",
    "select_patterns",
]

#: The two-sided floor annulus, in units of the pattern's rotor-mean carrier.
ANNULUS_FRAC = (0.45, 0.7)
#: Half-width of the peak search and of the line exclusion, in FFT bins.
PEAK_HALF_BINS = 1
EXCLUDE_HALF_BINS = 1
#: The prominence bars the audit reports. NONE of them is an established
#: threshold; three are reported so the reader can pick.
PROM_THRESHOLDS_DB = (3.0, 6.0, 10.0)
#: Orders the per-entry table names explicitly.
NAMED_ORDERS = (1, 2, 4, 8, 16)
#: ``profile_db`` value that switches the comb OFF: the line amplitude is
#: ``sqrt(2 * 10^(db/10))``, so -300 dB is 1e-30 of the power, thirty orders of
#: magnitude under any floor this model can hold.
COMB_OFF_DB = -300.0
#: Orders the bin geometry is precomputed for; entries are sliced out of it.
#: A standby payload carries 130 orders (its own fit reaches further up the
#: band at 35 rev/s than a cruise fit does at 80), so 256 clears every payload
#: this campaign can hand in.
K_GEOM = 256


# ---------------------------------------------------------------------------
# carrier patterns: what four real rotors were doing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pattern:
    """One four-rotor carrier pattern: a real window's per-rotor mean speed."""

    name: str
    rig: str
    regime: str
    support: str
    rev_s: tuple[float, ...]

    @property
    def mean_rev_s(self) -> float:
        return float(np.mean(self.rev_s))

    @property
    def spread_rev_s(self) -> float:
        return float(max(self.rev_s) - min(self.rev_s))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rig": self.rig,
            "regime": self.regime,
            "support": self.support,
            "rev_s": [round(float(v), 4) for v in self.rev_s],
            "mean_rev_s": round(self.mean_rev_s, 4),
            "spread_rev_s": round(self.spread_rev_s, 4),
        }


#: Which frozen window set feeds which pattern family, and how it is labelled.
PATTERN_SETS: tuple[tuple[str, str, str, str], ...] = (
    ("dregon", "cruise", "dregon", "dregon_score"),
    ("dregon", "cruise", "dregon", "dregon_fit"),
    ("michaels", "cruise", "michaels", "michaels_cruise"),
    ("michaels", "standby", "michaels", "michaels_standby"),
)


def select_patterns() -> list[Pattern]:
    """Two cruise patterns per rig plus two Michael's standby patterns.

    The rule is deterministic and measured, not taste: per family, the window
    with the NARROWEST and the one with the WIDEST per-rotor speed spread. The
    two bracket the only thing the pattern choice controls — how much the four
    rotors' combs interleave, and therefore how much of a rotor's local floor is
    its neighbours' lines.

    Reads the frozen supports through :mod:`experiments.noise_model.supports`;
    only the per-rotor frame-mean carrier of each window is used.
    """
    from experiments.noise_model import supports as SUP

    specs = RS._real_window_specs()
    families: dict[tuple[str, str], list[Pattern]] = {}
    for rig, regime, _label, tag in PATTERN_SETS:
        for spec in specs[tag]:
            sup = SUP.load_support(spec)
            rev = tuple(float(v) for v in sup.carrier_rev_s.mean(axis=1))
            families.setdefault((rig, regime), []).append(
                Pattern(name="", rig=rig, regime=regime, support=sup.name, rev_s=rev)
            )
    out: list[Pattern] = []
    for (rig, regime), items in families.items():
        ordered = sorted(items, key=lambda p: (p.spread_rev_s, p.support))
        for tag, pat in (("narrow", ordered[0]), ("wide", ordered[-1])):
            out.append(
                Pattern(
                    name=f"{rig}_{regime}_{tag}",
                    rig=rig,
                    regime=regime,
                    support=pat.support,
                    rev_s=pat.rev_s,
                )
            )
    return out


def patterns_from_dicts(rows: list[dict[str, Any]]) -> list[Pattern]:
    """Inverse of :meth:`Pattern.as_dict` — what a worker is handed."""
    return [
        Pattern(
            name=str(r["name"]),
            rig=str(r["rig"]),
            regime=str(r["regime"]),
            support=str(r["support"]),
            rev_s=tuple(float(v) for v in r["rev_s"]),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# bin geometry: peak bins, annulus bins, exclusions — once per pattern
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatternGeometry:
    """Every bin index the estimator needs, precomputed for one pattern.

    Built once and reused by every entry: the indices depend on the carrier
    pattern and the FFT grid alone, never on the payload.
    """

    centre: np.ndarray  # (R, K) nearest bin of k * f_r
    peak_idx: np.ndarray  # (R, K, 2 * PEAK_HALF_BINS + 1)
    ann_idx: np.ndarray  # (R, K, A) two-sided annulus bins
    ann_in_band: np.ndarray  # (R, K, A) inside the rfft grid
    ann_valid: np.ndarray  # (R, K, A) in band and not on a neighbouring line
    line_dist: np.ndarray  # (F,) distance in bins to the nearest rotor line
    k_geom: int

    @property
    def n_rotors(self) -> int:
        return int(self.centre.shape[0])


def _geometry(
    rev_s: np.ndarray, n_bins: int, df: float, *, k_geom: int = K_GEOM
) -> PatternGeometry:
    rev = np.asarray(rev_s, dtype=np.float64)
    n_rotors = int(rev.size)
    k = np.arange(1, k_geom + 1, dtype=np.float64)
    centre = np.rint(np.outer(rev, k) / df).astype(np.int64)  # (R, K)
    fbar = float(rev.mean())
    d_lo = int(np.ceil(ANNULUS_FRAC[0] * fbar / df))
    d_hi = int(np.floor(ANNULUS_FRAC[1] * fbar / df))
    d_lo = max(d_lo, PEAK_HALF_BINS + 1)
    if d_hi < d_lo:
        raise ValueError(f"annulus {ANNULUS_FRAC} of fbar={fbar:.3f} Hz holds no bin at df={df}")
    offs = np.arange(d_lo, d_hi + 1, dtype=np.int64)
    ann = np.concatenate([-offs[::-1], offs])  # (A,)
    ann_idx = centre[:, :, None] + ann[None, None, :]

    peak_off = np.arange(-PEAK_HALF_BINS, PEAK_HALF_BINS + 1, dtype=np.int64)
    peak_idx = centre[:, :, None] + peak_off[None, None, :]

    # the neighbouring-line exclusion: every rotor's k-1 / k / k+1 lines, +-1 bin
    pad = np.concatenate([np.zeros((n_rotors, 1), dtype=np.int64), centre], axis=1)  # k = 0..K
    nb = np.stack(
        [
            np.concatenate([pad[:, :1], centre[:, :-1]], axis=1),  # k - 1 (k=1 -> 0 Hz)
            centre,
            np.concatenate([centre[:, 1:], centre[:, -1:] + np.rint(rev[:, None] / df)], axis=1),
        ],
        axis=-1,
    )  # (R, K, 3)
    blocked = (
        nb.transpose(1, 0, 2).reshape(1, k_geom, n_rotors * 3)[:, :, :, None]
        + np.arange(-EXCLUDE_HALF_BINS, EXCLUDE_HALF_BINS + 1, dtype=np.int64)[None, None, None, :]
    ).reshape(1, k_geom, -1)  # (1, K, 9R)
    in_band = (ann_idx >= 0) & (ann_idx < n_bins) & (centre[:, :, None] < n_bins)
    hits_line = (ann_idx[:, :, :, None] == blocked[:, :, None, :]).any(axis=-1)
    ann_valid = in_band & ~hits_line

    # distance, in bins, from every bin to the nearest rotor line of any order
    all_lines = np.unique(centre[centre < n_bins])
    line_dist = np.abs(np.arange(n_bins)[:, None] - all_lines[None, :]).min(axis=1)
    return PatternGeometry(
        centre=np.clip(centre, 0, n_bins - 1),
        peak_idx=np.clip(peak_idx, 0, n_bins - 1),
        ann_idx=np.clip(ann_idx, 0, n_bins - 1),
        ann_in_band=in_band,
        ann_valid=ann_valid,
        line_dist=line_dist,
        k_geom=int(k_geom),
    )


# ---------------------------------------------------------------------------
# the probe: one expected periodogram per (payload, pattern)
# ---------------------------------------------------------------------------


class PatternProbe:
    """:class:`rig_sampler.ModelProbe` with a per-ROTOR carrier row.

    ``flight_rate_work`` already takes an ``(n_rotors, n_samples)`` label, so a
    pattern costs exactly what a single-speed probe costs: ~0.2 s per payload on
    one thread. The grid and each pattern's work-grid carrier are built once.
    """

    def __init__(self, patterns: list[Pattern], *, n_mics: int = 8) -> None:
        import torch

        from experiments.noise_model import spectrum as SP

        torch.set_num_threads(1)
        self.n_mics = int(n_mics)
        self.patterns = {p.name: p for p in patterns}
        self.grid = SP.flight_grid()
        self.freqs_hz = np.fft.rfftfreq(SP.FLIGHT_N_FFT, d=1.0 / float(SP.FLIGHT_SR))
        self.df = float(self.freqs_hz[1] - self.freqs_hz[0])
        self._rate = {
            p.name: SP.flight_rate_work(
                self.grid,
                np.repeat(np.asarray(p.rev_s, dtype=np.float64)[:, None], SP.FLIGHT_N_FFT, axis=1),
                np.array([0], dtype=np.int64),
            )
            for p in patterns
        }
        self.geom = {
            p.name: _geometry(np.asarray(p.rev_s, dtype=np.float64), self.freqs_hz.size, self.df)
            for p in patterns
        }

    def k_max(self, fit: dict[str, Any], pattern: str) -> int:
        from experiments.noise_model import spectrum as SP

        n_orders = int(np.asarray(fit["params"]["profile"]["profile_db"]).shape[1])
        fastest = float(max(self.patterns[pattern].rev_s))
        return min(
            n_orders,
            SP.k_max_for_carrier(fastest, SP.FLIGHT_SR, k_cap=n_orders),
            self.geom[pattern].k_geom,
        )

    def periodogram(
        self, fit: dict[str, Any], pattern: str, *, comb: bool = True, full_orders: bool = False
    ) -> np.ndarray:
        """``(M, F)`` expected periodogram; ``comb=False`` zeroes the lines.

        With the comb off the profile is a SINGLE -300 dB order unless
        ``full_orders``, because 130 orders of 1e-30 cost 0.2 s and add
        nothing; :func:`comb_switch_check` is what proves that.
        """
        import torch

        from experiments.noise_model import model as MD
        from experiments.noise_model import spectrum as SP

        p = fit["params"]
        k_max = self.k_max(fit, pattern)
        if not comb:
            prof_shape = np.asarray(p["profile"]["profile_db"], dtype=np.float64).shape
            k_max = k_max if full_orders else 1
            p = dict(p)
            prof = dict(p["profile"])
            prof["profile_db"] = np.full((prof_shape[0], k_max), COMB_OFF_DB)
            p["profile"] = prof
        with torch.no_grad():
            m = SP.flight_model(
                self.grid, MD.params_from_dict(p), rate_work=self._rate[pattern], k_max=k_max
            )
        return m.cpu().numpy()[: self.n_mics, 0, :]


# ---------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------


def local_floor(
    power: np.ndarray, geom: PatternGeometry, k_max: int
) -> tuple[np.ndarray, np.ndarray]:
    """``((M, R, K) local floor power, (R, K) fell-back mask)``.

    The annulus median with the neighbouring-line exclusion applied. Where the
    exclusion empties the annulus the UNEXCLUDED annulus is used instead and
    the order is flagged: at 7.8 Hz per bin a 35 rev/s standby carrier puts its
    orders 4.6 bins apart, so the 0.45-0.7 fbar annulus is two bins wide and
    both of them are within one bin of a neighbouring line. Falling back keeps
    the statistic defined at standby, at the cost of reading a floor that is
    mostly the neighbours' skirts — which the fallback count makes visible.
    """
    ann = power[:, geom.ann_idx[:, :k_max]]  # (M, R, K, A)
    valid = geom.ann_valid[None, :, :k_max]
    in_band = geom.ann_in_band[None, :, :k_max]
    empty = np.asarray(~geom.ann_valid[:, :k_max].any(axis=-1), dtype=bool)  # (R, K)
    use = np.where(empty[None, :, :, None], in_band, valid)
    with np.errstate(invalid="ignore"):
        floor = np.asarray(np.nanmedian(np.where(use, ann, np.nan), axis=-1))
    return floor, empty


def prominence_db(power: np.ndarray, geom: PatternGeometry, k_max: int) -> np.ndarray:
    """``(M, R, K)`` line-to-local-floor prominence, dB — the R4 estimator."""
    peak = power[:, geom.peak_idx[:, :k_max]].max(axis=-1)  # (M, R, K)
    floor, _empty = local_floor(power, geom, k_max)
    return 10.0 * np.log10(np.maximum(peak, 1e-300) / np.maximum(floor, 1e-300))


def line_level_db(power: np.ndarray, geom: PatternGeometry, k_max: int) -> np.ndarray:
    """``(M, R, K)`` peak level at each order's carrier bin, dB."""
    return 10.0 * np.log10(np.maximum(power[:, geom.peak_idx[:, :k_max]].max(axis=-1), 1e-300))


def floor_at_lines_db(power: np.ndarray, geom: PatternGeometry, k_max: int) -> np.ndarray:
    """``(M, R, K)`` comb-free floor level AT each order's carrier bin, dB."""
    return 10.0 * np.log10(np.maximum(power[:, geom.centre[:, :k_max]], 1e-300))


def comb_pedestal_db(
    p_tot: np.ndarray, p_floor: np.ndarray, geom: PatternGeometry
) -> dict[str, float]:
    """How much the COMB raises the spectrum where no line sits, in dB.

    Measured over the bins FARTHEST from any rotor line: 2 bins or more where
    the grid allows it (every cruise pattern), and otherwise the largest
    separation the pattern has (a 35 rev/s standby carrier puts its orders 4.6
    bins apart at 7.8 Hz per bin, so no bin is ever more than 2 away and often
    not more than 1). It is the part of the R4 "local floor" that is not the
    broadband floor at all: the four combs' merged skirts. It is why a
    prominence read against the local floor is several dB smaller than the same
    line read against the broadband floor, and it is a property of the model at
    its own 2048-point front end, not an artefact of the estimator.
    """
    reach = int(min(2, int(geom.line_dist.max())))
    mask = geom.line_dist >= reach
    d = 10.0 * np.log10(np.maximum(p_tot[:, mask], 1e-300) / np.maximum(p_floor[:, mask], 1e-300))
    return {
        "min_dist_bins": reach,
        "n_bins": int(mask.sum()),
        "median_db": float(np.median(d)),
        "p95_db": float(np.percentile(d, 95.0)),
        "max_db": float(np.max(d)),
        "min_db": float(np.min(d)),
    }


def comb_switch_check(fit: dict[str, Any], probe: PatternProbe, pattern: str) -> dict[str, float]:
    """Verify the comb switch: ``profile_db = -300`` at FULL ``k_max`` must
    equal the one-order shortcut the audit uses, bin for bin."""
    slow = probe.periodogram(fit, pattern, comb=False, full_orders=True)
    fast = probe.periodogram(fit, pattern, comb=False)
    d = 10.0 * np.log10(np.maximum(slow, 1e-300) / np.maximum(fast, 1e-300))
    return {"max_abs_db": float(np.max(np.abs(d))), "n_bins": int(slow.shape[1])}


# ---------------------------------------------------------------------------
# per-payload statistics
# ---------------------------------------------------------------------------


def profile_trend(profile_db: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(trend (R, K), gain (R,), slope (R,))`` from :func:`rig_sampler.decompose`."""
    prof = np.atleast_2d(np.asarray(profile_db, dtype=np.float64))
    parts = RS.decompose_block(prof)
    trend = np.stack([p.gain + p.slope * (p.x - p.x.mean()) for p in parts])
    return trend, np.array([p.gain for p in parts]), np.array([p.slope for p in parts])


def _quant(v: np.ndarray) -> list[float]:
    return [float(np.min(v)), float(np.median(v)), float(np.max(v))]


def payload_stats(
    fit: dict[str, Any], probe: PatternProbe, pattern: str, *, detail: bool = False
) -> tuple[dict[str, Any], np.ndarray]:
    """Every per-pattern number of ONE payload, plus its prominence curve.

    Two floors are read, and they answer different halves of the question:

    * ``prom_*`` — the R4 LOCAL floor (annulus median of ``P_tot``): what a
      separator sees around the line, the neighbours' skirts included. This is
      the headline family, and the only one the counts are built on.
    * ``prom_bb_k*`` — the same peak over the COMB-FREE broadband floor at the
      same bin, at the named orders only. It says how far the line would stand
      over the floor the model fitted if the rest of the comb were silent. It
      is NOT counted over orders, because at high ``k`` the peak bin holds the
      other orders' merged skirts (the pedestal) and the statistic saturates
      there at the pedestal instead of following the line.

    Returns ``(row, curve)`` where ``curve[k - 1]`` is the rotor-mean of the
    mic-MEDIAN local-floor prominence at order ``k`` — what the ladder draws.
    """
    geom = probe.geom[pattern]
    k_max = probe.k_max(fit, pattern)
    p_tot = probe.periodogram(fit, pattern, comb=True)
    p_floor = probe.periodogram(fit, pattern, comb=False)

    peak_db = line_level_db(p_tot, geom, k_max)  # (M, R, K)
    bb_db = floor_at_lines_db(p_floor, geom, k_max)  # (M, R, K)
    floor_pow, fell_back = local_floor(p_tot, geom, k_max)
    prom = peak_db - 10.0 * np.log10(np.maximum(floor_pow, 1e-300))
    promf = peak_db - bb_db

    prof_full = np.asarray(fit["params"]["profile"]["profile_db"], dtype=np.float64)
    trend, _gain, _slope = profile_trend(prof_full)
    resid = prof_full[:, :k_max] - trend[:, :k_max]
    line = np.maximum(10.0 ** (peak_db / 10.0) - 10.0 ** (bb_db / 10.0), 1e-300)
    # the SMOOTH trend's line (residual wiggle removed) over the broadband floor
    trend_med = np.nanmedian(10.0 * np.log10(line) - resid[None, :, :] - bb_db, axis=0)  # (R, K)

    ks = np.arange(1, k_max + 1)
    row: dict[str, Any] = {"k_max": int(k_max)}
    per_rotor = np.nanmedian(prom, axis=0)  # (R, K) mic-median
    per_mic = np.nanmean(prom, axis=1)  # (M, K) rotor-mean
    bb_rotor = np.nanmean(promf, axis=1)  # (M, K) rotor-mean, broadband floor
    for k in NAMED_ORDERS:
        if k > k_max:
            continue
        row[f"prom_k{k}"] = _quant(per_mic[:, k - 1])
        row[f"prom_bb_k{k}"] = _quant(bb_rotor[:, k - 1])
    for thr in PROM_THRESHOLDS_DB:
        hits = per_rotor >= thr
        counts = hits.sum(axis=1).astype(np.float64)
        highest = np.array([int(ks[h].max()) if h.any() else 0 for h in hits], dtype=np.float64)
        bar = f"{thr:g}".replace(".", "p")
        row[f"count_ge{bar}"] = [float(counts.mean()), *_quant(counts)]
        row[f"highest_ge{bar}"] = [float(highest.mean()), *_quant(highest)]
    curve = np.nanmean(per_rotor, axis=0).astype(np.float32)
    row["n_nan_orders"] = int(np.isnan(per_rotor).sum())
    # the order from which the SMOOTH trend stays under the floor: the last order
    # above it plus one, not the first dip (the floor's own low-frequency tilt
    # can put k=1 under it while k=2..20 stand well clear)
    above = trend_med >= 0.0
    cross = np.array([int(ks[a].max()) + 1 if a.any() else 1 for a in above], dtype=np.float64)
    row["trend_cross_order"] = [float(cross.mean()), *_quant(cross)]
    ped = comb_pedestal_db(p_tot, p_floor, geom)
    row["pedestal"] = ped if detail else round(ped["median_db"], 3)
    row["n_fallback_orders"] = int(fell_back.sum())
    return row, curve


def payload_shape(fit: dict[str, Any]) -> dict[str, Any]:
    """The pattern-independent coordinates of one payload."""
    prof = np.asarray(fit["params"]["profile"]["profile_db"], dtype=np.float64)
    _trend, gain, slope = profile_trend(prof)
    n_k = int(prof.shape[1])
    drop = -slope * np.log10(float(n_k))
    floor = fit["params"]["floor"]
    return {
        "n_orders": n_k,
        "rotor_gain_db": [float(gain.mean()), *_quant(gain)],
        "slope_db_dec": [float(slope.mean()), *_quant(slope)],
        "trend_drop_db": [float(drop.mean()), *_quant(drop)],
        "floor_mean_db": float(floor["floor_mean_db"]),
        # a noise-v3-fit/1 floor is the spline alone: no tilt site, i.e. zero
        "floor_tilt_db_oct": float(floor.get("floor_tilt_db_oct", 0.0)),
        "gain_minus_floor_db": float(gain.mean() - float(floor["floor_mean_db"])),
    }


def entry_row(
    payloads: dict[str, dict[str, Any]],
    probe: PatternProbe,
    patterns: list[str],
    *,
    detail: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Audit one entry: ``{"cruise": fit, "standby": fit | None}``.

    Cruise patterns are measured on the cruise payload and standby patterns on
    the standby payload, which is the regime composition the renderer uses.
    """
    row: dict[str, Any] = {"shape": payload_shape(payloads["cruise"])}
    if payloads.get("standby") is not None:
        row["shape_standby"] = payload_shape(payloads["standby"])
    curves: dict[str, np.ndarray] = {}
    for name in patterns:
        regime = probe.patterns[name].regime
        fit = payloads["cruise"] if regime == "cruise" else payloads.get("standby")
        if fit is None:
            continue
        stats, curve = payload_stats(fit, probe, name, detail=detail)
        row[name] = stats
        curves[name] = curve
    return row, curves


# ---------------------------------------------------------------------------
# the real references
# ---------------------------------------------------------------------------


def real_profile_slopes() -> list[dict[str, Any]]:
    """The 16 MEASURED v2 rotor profiles' trend slope and total drop.

    Two cruise rigs x 4 rotors, Michael's standby x 4 rotors, and the four
    DREGON bench motors at throttle 70 — the same 16 profiles
    ``rig_sampler``'s trend-guard calibration is measured over.
    """
    out: list[dict[str, Any]] = []
    sources: list[tuple[str, str]] = [
        ("dregon_cruise", RS.ANCHORS["dregon"]["cruise"]),
        ("michaels_cruise", RS.ANCHORS["michaels"]["cruise"]),
        ("michaels_standby", RS.ANCHORS["michaels"]["standby"]),
    ]
    sources += [(f"bench_motor{i + 1}", p) for i, p in enumerate(RS.BENCH_MOTORS)]
    for name, path in sources:
        fit = RS.load_fit(path)
        prof = np.atleast_2d(np.asarray(fit["params"]["profile"]["profile_db"], dtype=np.float64))
        for r, parts in enumerate(RS.decompose_block(prof)):
            out.append(
                {
                    "profile": f"{name}_r{r}",
                    "n_orders": int(prof.shape[1]),
                    "slope_db_dec": float(parts.slope),
                    "trend_drop_db": float(-parts.slope * np.log10(float(prof.shape[1]))),
                    "gain_db": float(parts.gain),
                }
            )
    return out
