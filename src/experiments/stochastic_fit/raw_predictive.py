"""Non-circular posterior-predictive checks on real and rendered waveforms.

No candidate model is fitted inside this module. The same carrier-conditioned,
local-floor statistic is extracted directly from held-out real audio and from
posterior-predictive audio rendered on the identical RPS trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import Clip, periodogram
from .predictive import ORDER_POINTS, VISIBLE_THRESHOLDS_DB, two_sample_classifier

AUC_UPPER_LIMIT = 0.80


@dataclass(frozen=True)
class RawTopologyDraws:
    """Model-free line/local-floor margins from a collection of waveforms."""

    margin_db: np.ndarray  # (clips, rotors, orders); NaN where attribution is impossible
    observations: np.ndarray  # same shape; independent-enough frame count
    carrier_id: np.ndarray | None = None  # (clips,), clusters repeated matched-RPS draws

    def __post_init__(self) -> None:
        margin = np.asarray(self.margin_db)
        observations = np.asarray(self.observations)
        if margin.ndim != 3 or observations.shape != margin.shape:
            raise ValueError("raw topology arrays must share shape (clips, rotors, orders)")
        if np.isinf(margin).any() or not np.isfinite(observations).all():
            raise ValueError("raw topology arrays contain invalid values")
        if self.carrier_id is not None and np.asarray(self.carrier_id).shape != (margin.shape[0],):
            raise ValueError("carrier_id must have one entry per clip")


def line_floor_margins(
    clip: Clip,
    *,
    k_max: int = 64,
    min_rps: float = 5.0,
    line_half_bins: int = 2,
    guard_bins: int = 2,
    frame_stride: int = 2,
    censor_db: float = -40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-order union-comb margins measured directly from one waveform.

    The four rotors are deliberately aggregated: on DREGON their nearby lines
    are rarely attributable to one rotor without fitting a candidate model.
    For each order this measures the union of all telemetry-predicted line
    windows against masked neighbouring bins. Non-positive excess is retained
    at a fixed censoring level rather than dropping invisible teeth.
    """
    pg = periodogram(clip)
    power = np.asarray(pg.power, dtype=np.float64)
    n_rotors = pg.rps.shape[0]
    values: list[list[float]] = [[] for _ in range(k_max)]
    df = pg.df
    n_freq = pg.freqs.size
    orders = np.arange(1, k_max + 1, dtype=np.float64)
    for frame in range(0, pg.rps.shape[1], frame_stride):
        by_order: list[list[tuple[int, float]]] = [[] for _ in range(k_max)]
        all_bins: list[int] = []
        for rotor in range(n_rotors):
            speed = float(pg.rps[rotor, frame])
            if speed < min_rps:
                continue
            centres = orders * speed
            live = np.flatnonzero((centres >= pg.freqs[1]) & (centres < pg.freqs[-1]))
            for order_index in live:
                bin_index = int(
                    np.clip(
                        round(float(centres[order_index]) / df),
                        1,
                        n_freq - 2,
                    )
                )
                by_order[int(order_index)].append((bin_index, speed))
                all_bins.append(bin_index)
        if not all_bins:
            continue
        global_line_mask = np.zeros(n_freq, dtype=bool)
        for bin_index in all_bins:
            lo = max(0, bin_index - guard_bins)
            hi = min(n_freq, bin_index + guard_bins + 1)
            global_line_mask[lo:hi] = True

        for order_index, targets in enumerate(by_order):
            if not targets:
                continue
            target_mask = np.zeros(n_freq, dtype=bool)
            neighbourhood = np.zeros(n_freq, dtype=bool)
            for bin_index, speed in targets:
                line_lo = max(0, bin_index - line_half_bins)
                line_hi = min(n_freq, bin_index + line_half_bins + 1)
                target_mask[line_lo:line_hi] = True
                half_width = max(
                    int(round(0.45 * speed / df)),
                    guard_bins + 4,
                )
                floor_lo = max(1, bin_index - half_width)
                floor_hi = min(n_freq, bin_index + half_width + 1)
                neighbourhood[floor_lo:floor_hi] = True
            line_bins = np.flatnonzero(target_mask)
            floor_bins = np.flatnonzero(neighbourhood & ~global_line_mask)
            masked_floor = True
            if floor_bins.size < 4:
                # At high comb density there may be no line-free bin. A lower
                # quantile of the same local neighbourhood is a symmetric
                # floor proxy for real and rendered audio.
                floor_bins = np.flatnonzero(neighbourhood & ~target_mask)
                masked_floor = False
            if line_bins.size == 0 or floor_bins.size < 4:
                continue
            local = power[:, frame, floor_bins]
            local_floor = (
                np.median(local, axis=1) if masked_floor else np.quantile(local, 0.25, axis=1)
            )
            local_floor = np.maximum(local_floor, 1e-20)
            line = power[:, frame, line_bins]
            excess = line.sum(axis=1) - local_floor * line_bins.size
            ratio = excess / (local_floor * line_bins.size)
            margin = 10.0 * np.log10(np.maximum(ratio, 10.0 ** (censor_db / 10.0)))
            values[order_index].append(float(np.median(margin)))

    margins = np.full((1, k_max), np.nan, dtype=np.float64)
    observations = np.zeros_like(margins)
    for order_index, cell in enumerate(values):
        if cell:
            margins[0, order_index] = float(np.median(cell))
            observations[0, order_index] = len(cell)
    return margins, observations


def extract_raw_topology(clips: list[Clip], *, k_max: int = 64) -> RawTopologyDraws:
    margins, observations = zip(
        *(line_floor_margins(clip, k_max=k_max) for clip in clips), strict=True
    )
    carrier_id = np.asarray(
        [clip.meta.get("matched_real_clip", clip.clip_id) for clip in clips],
        dtype=object,
    )
    return RawTopologyDraws(np.stack(margins), np.stack(observations), carrier_id)


def _finite_row_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    count = valid.sum(axis=1)
    safe = np.where(valid, values, 0.0)
    mean = safe.sum(axis=1) / np.maximum(count, 1)
    variance = np.where(valid, (values - mean[:, None]) ** 2, 0.0).sum(axis=1)
    std = np.sqrt(variance / np.maximum(count, 1))
    mean = np.where(count > 0, mean, -40.0)
    return mean, std, count / values.shape[1]


def raw_topology_features(draws: RawTopologyDraws) -> tuple[np.ndarray, list[str]]:
    margin = draws.margin_db
    n_clips, n_rotors, n_harmonics = margin.shape
    columns: list[np.ndarray] = []
    names: list[str] = []
    for order in ORDER_POINTS:
        if order > n_harmonics:
            continue
        mean, std, fraction = _finite_row_stats(margin[:, :, order - 1])
        columns.extend((mean, std, fraction))
        names.extend(
            (
                f"raw_margin_k{order}_mean",
                f"raw_margin_k{order}_rotor_std",
                f"raw_margin_k{order}_observed",
            )
        )

    valid = np.isfinite(margin)
    for threshold in VISIBLE_THRESHOLDS_DB:
        visible = np.where(valid, margin > threshold, False).sum(axis=2)
        columns.extend((visible.mean(axis=1), visible.std(axis=1)))
        names.extend(
            (f"raw_visible_gt{threshold:g}_mean", f"raw_visible_gt{threshold:g}_rotor_std")
        )
    observed = valid.sum(axis=2)
    columns.extend((observed.mean(axis=1), observed.std(axis=1)))
    names.extend(("raw_observed_orders_mean", "raw_observed_orders_rotor_std"))

    order_values = np.arange(1, n_harmonics + 1, dtype=np.float64)
    for label, lo, hi in (("low", 2, 8), ("mid", 9, 24), ("high", 25, n_harmonics)):
        keep = (order_values >= lo) & (order_values <= hi)
        slopes = np.full((n_clips, n_rotors), np.nan)
        x_all = np.log10(order_values[keep])
        for clip_index in range(n_clips):
            for rotor in range(n_rotors):
                y = margin[clip_index, rotor, keep]
                finite = np.isfinite(y)
                if finite.sum() < 2:
                    continue
                x = x_all[finite]
                x = x - x.mean()
                slopes[clip_index, rotor] = float(
                    np.sum(x * (y[finite] - y[finite].mean())) / np.sum(x * x)
                )
        mean, std, fraction = _finite_row_stats(slopes)
        columns.extend((mean, std, fraction))
        names.extend(
            (
                f"raw_slope_{label}_mean",
                f"raw_slope_{label}_rotor_std",
                f"raw_slope_{label}_observed",
            )
        )

    bpf = np.full((n_clips, n_rotors), np.nan)
    for clip_index in range(n_clips):
        for rotor in range(n_rotors):
            values = margin[clip_index, rotor, :4]
            if np.isfinite(values).all():
                bpf[clip_index, rotor] = values[1] - values[[0, 2, 3]].mean()
    mean, std, fraction = _finite_row_stats(bpf)
    columns.extend((mean, std, fraction))
    names.extend(
        (
            "raw_bpf_prominence_mean",
            "raw_bpf_prominence_rotor_std",
            "raw_bpf_prominence_observed",
        )
    )
    features = np.column_stack(columns).reshape(n_clips, -1)
    return features, names


def raw_topology_gate(
    real_train: RawTopologyDraws,
    real_test: RawTopologyDraws,
    synthetic_train: RawTopologyDraws,
    synthetic_test: RawTopologyDraws,
    *,
    seed: int = 0,
    n_bootstrap: int = 1000,
    run_classifier: bool = True,
) -> dict[str, Any]:
    train_real_x, names = raw_topology_features(real_train)
    test_real_x, names_test = raw_topology_features(real_test)
    train_synth_x, names_synth = raw_topology_features(synthetic_train)
    test_synth_x, names_synth_test = raw_topology_features(synthetic_test)
    if not (names == names_test == names_synth == names_synth_test):
        raise ValueError("raw topology collections must share one order grid")
    classifier = (
        two_sample_classifier(
            train_real_x,
            test_real_x,
            train_synth_x,
            test_synth_x,
            seed=seed,
            n_bootstrap=n_bootstrap,
            test_real_groups=real_test.carrier_id,
            test_synthetic_groups=synthetic_test.carrier_id,
        )
        if run_classifier
        else {"classifier_auc": None, "classifier_auc_95": None}
    )

    real = real_test.margin_db
    synthetic = synthetic_test.margin_db
    covered = 0
    total = 0
    real_curve: list[float] = []
    synthetic_curve: list[float] = []
    curve_scale: list[float] = []
    curve_rows: list[dict[str, Any]] = []
    for order_index in range(min(real.shape[2], synthetic.shape[2])):
        r = real[:, :, order_index]
        s = synthetic[:, :, order_index]
        r = r[np.isfinite(r)]
        s = s[np.isfinite(s)]
        if r.size == 0 or s.size < 4:
            continue
        lo, hi = np.quantile(s, (0.05, 0.95))
        covered += int(((r >= lo) & (r <= hi)).sum())
        total += r.size
        real_curve.append(float(np.median(r)))
        synthetic_curve.append(float(np.median(s)))
        curve_scale.append(max(float(np.subtract(*np.quantile(r, (0.75, 0.25)))), 1.0))
        curve_rows.append(
            {
                "order": order_index + 1,
                "real_n": int(r.size),
                "synthetic_n": int(s.size),
                "real_q10_median_q90": [float(value) for value in np.quantile(r, (0.1, 0.5, 0.9))],
                "synthetic_q10_median_q90": [
                    float(value) for value in np.quantile(s, (0.1, 0.5, 0.9))
                ],
            }
        )
    coverage = float(covered / total) if total else float("nan")
    curve_rmse = float(
        np.sqrt(
            np.mean(
                ((np.asarray(synthetic_curve) - np.asarray(real_curve)) / np.asarray(curve_scale))
                ** 2
            )
        )
    )
    feature_scale = np.maximum(test_real_x.std(axis=0), 1.0)
    feature_shift = np.abs(test_synth_x.mean(axis=0) - test_real_x.mean(axis=0)) / feature_scale
    top_feature_shifts = [
        {"feature": names[index], "real_std_shift": float(feature_shift[index])}
        for index in np.argsort(feature_shift)[::-1][:10]
    ]
    direct_ok = coverage >= 0.80 and curve_rmse <= 1.0
    if not run_classifier:
        status = "descriptive"
    else:
        point_value = classifier["classifier_auc"]
        interval = classifier["classifier_auc_95"]
        assert point_value is not None and isinstance(interval, list)
        point_auc = float(point_value)
        upper_auc = float(interval[1])
        if direct_ok and upper_auc < AUC_UPPER_LIMIT:
            status = "pass"
        elif point_auc > 0.70 or coverage < 0.65 or curve_rmse > 1.5:
            status = "fail"
        else:
            status = "inconclusive"
    return {
        "scope": "direct-waveform-posterior-predictive",
        "status": status,
        "passed": status == "pass",
        "classifier_enabled": run_classifier,
        **classifier,
        "classifier_auc_upper_limit": AUC_UPPER_LIMIT,
        "predictive_90_coverage": coverage,
        "median_curve_iqr_rmse": curve_rmse,
        "per_order_margin_db": curve_rows,
        "top_feature_shifts": top_feature_shifts,
        "feature_names": names,
        "real_observed_cells": int(np.isfinite(real).sum()),
        "synthetic_observed_cells": int(np.isfinite(synthetic).sum()),
    }


def _subset(draws: RawTopologyDraws, indices: np.ndarray) -> RawTopologyDraws:
    carrier_id = None if draws.carrier_id is None else np.asarray(draws.carrier_id)[indices]
    return RawTopologyDraws(
        draws.margin_db[indices],
        draws.observations[indices],
        carrier_id,
    )


def repeated_waveform_gate_control(
    train_pool: RawTopologyDraws,
    test_pool: RawTopologyDraws,
    *,
    synthetic_draws_per_carrier: int = 4,
    repeats: int = 20,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """False-rejection calibration for the exact one-vs-many Michael gate."""
    if train_pool.carrier_id is None or test_pool.carrier_id is None:
        raise ValueError("repeated gate control requires carrier IDs")
    if synthetic_draws_per_carrier < 1 or repeats < 1:
        raise ValueError("draw and repeat counts must be positive")
    rng = np.random.default_rng(seed)

    def split(pool: RawTopologyDraws) -> tuple[RawTopologyDraws, RawTopologyDraws]:
        real_indices: list[int] = []
        synthetic_indices: list[int] = []
        carrier_id = np.asarray(pool.carrier_id)
        for carrier in np.unique(carrier_id):
            available = np.flatnonzero(carrier_id == carrier)
            need = synthetic_draws_per_carrier + 1
            if available.size < need:
                raise ValueError(f"carrier {carrier!r} has {available.size} draws, need {need}")
            selected = rng.choice(available, need, replace=False)
            real_indices.append(int(selected[0]))
            synthetic_indices.extend(int(index) for index in selected[1:])
        return (
            _subset(pool, np.asarray(real_indices)),
            _subset(pool, np.asarray(synthetic_indices)),
        )

    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        train_real, train_synthetic = split(train_pool)
        test_real, test_synthetic = split(test_pool)
        result = raw_topology_gate(
            train_real,
            test_real,
            train_synthetic,
            test_synthetic,
            seed=seed + repeat + 1,
            n_bootstrap=n_bootstrap,
            run_classifier=True,
        )
        rows.append(
            {
                "status": result["status"],
                "auc": result["classifier_auc"],
                "auc_upper": result["classifier_auc_95"][1],
                "coverage": result["predictive_90_coverage"],
                "curve_rmse": result["median_curve_iqr_rmse"],
            }
        )
    passed = np.asarray([row["status"] == "pass" for row in rows])
    return {
        "repeats": repeats,
        "pass_rate": float(passed.mean()),
        "false_rejection_rate": float(1.0 - passed.mean()),
        "auc_median_q95": [
            float(value) for value in np.quantile([row["auc"] for row in rows], (0.5, 0.95))
        ],
        "auc_upper_median_q95": [
            float(value) for value in np.quantile([row["auc_upper"] for row in rows], (0.5, 0.95))
        ],
        "coverage_median_q05": [
            float(value) for value in np.quantile([row["coverage"] for row in rows], (0.5, 0.05))
        ],
        "curve_rmse_median_q95": [
            float(value) for value in np.quantile([row["curve_rmse"] for row in rows], (0.5, 0.95))
        ],
        "rows": rows,
    }


def population_ranges(summary: dict[str, Any], base_ranges: dict[str, Any]) -> dict[str, Any]:
    """Inject fitted line and floor populations into a renderer preset."""
    rig = summary["rig"]
    population = summary["population"]
    example = next(iter(summary["train"].values()))["params"]
    floor_shape = np.asarray(example["floor_shape_db"], dtype=np.float64)
    floor_hz = np.asarray(example["floor_ctrl_hz"], dtype=np.float64)
    # The renderer centres its preset control points; compensate in the
    # line/floor base ratio so the realized local floor is unchanged.
    shape_mean = float(floor_shape.mean())
    width_scale = np.asarray(rig["width_scale"], dtype=np.float64)
    gamma0 = float(rig["gamma0"]) * width_scale
    gamma_slope = float(rig["gamma_slope"]) * width_scale
    # A Gaussian line with shaft-rate std sigma has HWHM
    # sqrt(2 log 2) * k * sigma. The fitted intercept is dominated by the
    # analysis-window floor/carrier nuisance and has no FM counterpart; the
    # identifiable high-order slope transfers exactly.
    shaft_jitter = gamma_slope / np.sqrt(2.0 * np.log(2.0))
    ranges = dict(base_ranges)
    ranges.update(
        profile_mean_db=rig["profile_db"],
        profile_basis_db=rig.get("profile_basis_db"),
        profile_rotor_db=rig["delta_db"],
        profile_residual_std_db=summary.get("profile_residual_std_db"),
        line_floor_mean_db=population["line_floor_mean_db"] - shape_mean,
        line_floor_std_db=population["line_floor_std_db"],
        rotor_contrast_std_db=population["rotor_contrast_std_db"],
        fixed_gamma0_hz=gamma0.tolist(),
        fixed_gamma_slope_hz=gamma_slope.tolist(),
        fixed_shaft_jitter_rps=shaft_jitter.tolist(),
        fixed_mic_gain_db=rig["mic_gain_db"],
        fixed_mic_floor_db=rig["mic_floor_db"],
        fixed_mic_gain_all_db=rig["gain_all_db"],
        floor_shape_preset=[
            [float(hz), float(db)] for hz, db in zip(floor_hz, floor_shape, strict=True)
        ],
        floor_shape_std_db=[0.0, 0.0],
        floor_tilt_db_oct=[
            float(example["floor_tilt_db_oct"]),
            float(example["floor_tilt_db_oct"]),
        ],
    )
    dynamics = summary.get("dynamics")
    if dynamics is not None:
        components = dynamics["components"]
        if len(components) != 1:
            raise ValueError(
                "the renderer carries ONE amplitude timescale; the envelope fit selected "
                f"{len(components)} components, which needs a renderer extension rather "
                "than a silently collapsed transfer"
            )
        component = components[0]
        # Degenerate ranges: these are measured, not prior beliefs to draw from.
        ranges.update(
            harm_gp_kernel="ou",  # the fitted density is Lorentzian
            harm_gp_std_db=[component["std_db"], component["std_db"]],
            harm_gp_tau_s=[component["tau_s"], component["tau_s"]],
            harm_coherence=[component["coherence"], component["coherence"]],
        )
    visibility = summary.get("visibility_model")
    if visibility is not None:
        ranges.update(
            visibility_probability=visibility["visible_probability"],
            visibility_attenuation_db=visibility["attenuation_mean_db"],
            visibility_attenuation_std_db=visibility["attenuation_std_db"],
        )
    return ranges


def render_matched_population(
    summary: dict[str, Any],
    clips: list[Clip],
    *,
    base_ranges: dict[str, Any],
    line_mode: str,
    mic_gain_db: tuple[float, float],
    seed: int,
    observation_augmentations: list[dict[str, Any]] | None = None,
    draws_per_clip: int = 1,
) -> list[Clip]:
    """Render one independent population draw on every real RPS trajectory."""
    from data_processing.stochastic_rotor_noise import (
        StochasticRanges,
        sample_params,
        synthesize,
    )

    if observation_augmentations:
        import tdseries as td

        from data_processing.noise_augmentations import (
            maybe_apply_noise_augmentation,
        )

    ranges = StochasticRanges.from_dict(population_ranges(summary, base_ranges))
    n_harmonics = len(summary["rig"]["profile_db"])
    rendered: list[Clip] = []
    if draws_per_clip < 1:
        raise ValueError("draws_per_clip must be positive")
    for clip_index, clip in enumerate(clips):
        if clip.sr != 16000:
            raise ValueError(f"population renderer expects 16 kHz, got {clip.sr}")
        for draw_index in range(draws_per_clip):
            draw_seed = seed + clip_index * draws_per_clip + draw_index
            rng = np.random.default_rng(draw_seed)
            params = sample_params(
                rng,
                ranges,
                n_rotors=clip.rps.shape[0],
                n_harmonics=n_harmonics,
                sample_rate=clip.sr,
            )
            params = params.with_(
                amp_rps_exponent=float(summary["rig"]["amp_exp"]),
                amp_rps_exponent_floor=float(summary["rig"]["floor_exp"]),
                floor_static_rel=float(summary["rig"]["floor_static_rel"]),
            )
            audio, _ = synthesize(
                params,
                clip.rps,
                rng=rng,
                n_mics=clip.audio.shape[0],
                mic_gain_db=mic_gain_db,
                line_mode=line_mode,
            )
            if observation_augmentations:
                frame = td.Frame(
                    {
                        "audio": td.uniform(
                            np.asarray(audio, dtype=np.float32),
                            clip.sr,
                            dims=("channel", "time"),
                        ),
                        "rps": td.uniform(
                            np.asarray(clip.rps, dtype=np.float32),
                            clip.sr,
                            dims=("rotor", "time"),
                        ),
                    }
                )
                for augmentation in observation_augmentations:
                    frame = maybe_apply_noise_augmentation(
                        frame,
                        augmentation,
                        rng,
                        target_len=clip.audio.shape[1],
                        sample_rate=clip.sr,
                    )
                audio = np.asarray(frame["audio"].data, dtype=np.float32)
            rendered.append(
                Clip(
                    f"posterior_{clip.clip_id}_{draw_seed}",
                    "posterior_predictive",
                    np.asarray(audio, dtype=np.float32),
                    clip.rps.copy(),
                    clip.sr,
                    clip.rps.copy(),
                    {"matched_real_clip": clip.clip_id, "seed": draw_seed},
                )
            )
    return rendered


__all__ = [
    "AUC_UPPER_LIMIT",
    "RawTopologyDraws",
    "extract_raw_topology",
    "line_floor_margins",
    "population_ranges",
    "raw_topology_features",
    "raw_topology_gate",
    "repeated_waveform_gate_control",
    "render_matched_population",
]
