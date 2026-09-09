"""Held-out posterior-predictive checks for the rig population.

The first gate targets the defect exposed by listening: joint comb topology.
One row represents one clip, so the classifier cannot leak four correlated
rotors across train and test. It trains on one real recording and is evaluated
on the other; synthetic draws use independent seeds in the two halves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

ORDER_POINTS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
VISIBLE_THRESHOLDS_DB = (0.0, 6.0, 12.0)


@dataclass(frozen=True)
class ProfileDraws:
    """Scale-aware line populations on one common order grid."""

    profile_db: np.ndarray  # (clips, rotors, orders), unit-area line power
    floor_db: np.ndarray  # (clips, orders), local floor PSD at 80 rev/s
    gamma_hz: np.ndarray  # (clips, rotors, orders), HWHM

    def __post_init__(self) -> None:
        profile = np.asarray(self.profile_db)
        floor = np.asarray(self.floor_db)
        gamma = np.asarray(self.gamma_hz)
        if profile.ndim != 3:
            raise ValueError(f"profile_db must be (C,R,K), got {profile.shape}")
        if floor.shape != (profile.shape[0], profile.shape[2]):
            raise ValueError("floor_db must be (C,K) on the profile order grid")
        if gamma.shape != profile.shape:
            raise ValueError("gamma_hz must have the profile shape")
        if profile.shape[2] < 4:
            raise ValueError("topology gate requires at least four orders")
        if not (
            np.isfinite(profile).all()
            and np.isfinite(floor).all()
            and np.isfinite(gamma).all()
            and (gamma > 0).all()
        ):
            raise ValueError("profile draws must be finite with positive widths")

    @property
    def peak_margin_db(self) -> np.ndarray:
        peak = self.profile_db - 10.0 * np.log10(np.pi * self.gamma_hz)
        return peak - self.floor_db[:, None, :]


def _local_floor(
    *,
    floor_mean_db: float,
    floor_shape_db: np.ndarray,
    floor_ctrl_hz: np.ndarray,
    floor_tilt_db_oct: float,
    n_harmonics: int,
    ref_rps: float = 80.0,
) -> np.ndarray:
    order_hz = ref_rps * np.arange(1, n_harmonics + 1, dtype=np.float64)
    shape = np.interp(
        np.log2(order_hz),
        np.log2(np.asarray(floor_ctrl_hz, dtype=np.float64)),
        np.asarray(floor_shape_db, dtype=np.float64),
    )
    tilt = floor_tilt_db_oct * np.log2(np.maximum(order_hz, 1.0) / 500.0)
    return floor_mean_db + shape + tilt


def fitted_profile_draws(summary: dict[str, Any], split: str) -> ProfileDraws:
    clips = summary.get(split)
    if not clips:
        raise ValueError(f"population summary has no {split!r} clips")
    profiles: list[np.ndarray] = []
    floors: list[np.ndarray] = []
    gammas: list[np.ndarray] = []
    for clip in clips.values():
        params = clip["params"]
        profile = np.asarray(params["profile_db"], dtype=np.float64)
        gamma = np.asarray(params["gamma"], dtype=np.float64)
        n_harmonics = profile.shape[1]
        profiles.append(profile)
        gammas.append(gamma)
        floors.append(
            _local_floor(
                floor_mean_db=float(params["floor_mean_db"]),
                floor_shape_db=np.asarray(params["floor_shape_db"]),
                floor_ctrl_hz=np.asarray(params["floor_ctrl_hz"]),
                floor_tilt_db_oct=float(params["floor_tilt_db_oct"]),
                n_harmonics=n_harmonics,
            )
        )
    k = min(profile.shape[1] for profile in profiles)
    return ProfileDraws(
        profile_db=np.stack([profile[:, :k] for profile in profiles]),
        floor_db=np.stack([floor[:k] for floor in floors]),
        gamma_hz=np.stack([gamma[:, :k] for gamma in gammas]),
    )


def synthetic_profile_draws(summary: dict[str, Any], n_clips: int, *, seed: int) -> ProfileDraws:
    """Draw exactly the profile population exported by ``fit_population``."""
    if n_clips < 1:
        raise ValueError("n_clips must be positive")
    rng = np.random.default_rng(seed)
    rig = summary["rig"]
    population = summary["population"]
    mean = np.asarray(rig["profile_db"], dtype=np.float64)
    rotor = np.asarray(rig["delta_db"], dtype=np.float64)
    basis_value = rig.get("profile_basis_db")
    basis = (
        np.zeros((0, mean.size), dtype=np.float64)
        if basis_value is None
        else np.asarray(basis_value, dtype=np.float64)
    )
    n_rotors, n_harmonics = rotor.shape
    factors = rng.standard_normal((n_clips, n_rotors, basis.shape[0]))
    contrast = rng.normal(
        0.0,
        float(population["rotor_contrast_std_db"]),
        (n_clips, n_rotors),
    )
    contrast -= contrast.mean(axis=1, keepdims=True)
    profile = (
        mean[None, None, :]
        + rotor[None, :, :]
        + np.einsum("crq,qk->crk", factors, basis)
        + contrast[:, :, None]
    )
    ratio = rng.normal(
        float(population["line_floor_mean_db"]),
        float(population["line_floor_std_db"]),
        n_clips,
    )
    reference_order = min(1, n_harmonics - 1)
    floor_mean = profile[:, :, reference_order].mean(axis=1) - ratio

    example = next(iter(summary["train"].values()))["params"]
    floor = np.stack(
        [
            _local_floor(
                floor_mean_db=float(value),
                floor_shape_db=np.asarray(example["floor_shape_db"]),
                floor_ctrl_hz=np.asarray(example["floor_ctrl_hz"]),
                floor_tilt_db_oct=float(example["floor_tilt_db_oct"]),
                n_harmonics=n_harmonics,
            )
            for value in floor_mean
        ]
    )
    width_scale = np.asarray(rig["width_scale"], dtype=np.float64)
    orders = np.arange(1, n_harmonics + 1, dtype=np.float64)
    gamma = (float(rig["gamma0"]) + float(rig["gamma_slope"]) * orders)[None, :] * width_scale[
        :, None
    ]
    return ProfileDraws(
        profile_db=profile,
        floor_db=floor,
        gamma_hz=np.broadcast_to(gamma, profile.shape).copy(),
    )


def topology_features(draws: ProfileDraws) -> tuple[np.ndarray, list[str]]:
    """Clip-level features; every rotor is aggregated inside its clip."""
    margin = draws.peak_margin_db
    n_clips, _, n_harmonics = margin.shape
    order_indices = np.array(
        [order - 1 for order in ORDER_POINTS if order <= n_harmonics], dtype=int
    )
    columns: list[np.ndarray] = []
    names: list[str] = []
    for index in order_indices:
        order = index + 1
        columns.extend(
            (np.median(margin[:, :, index], axis=1), np.std(margin[:, :, index], axis=1))
        )
        names.extend((f"margin_k{order}_median", f"margin_k{order}_rotor_std"))

    for threshold in VISIBLE_THRESHOLDS_DB:
        count = (margin > threshold).sum(axis=2)
        columns.extend((count.mean(axis=1), count.std(axis=1)))
        names.extend((f"visible_gt{threshold:g}_mean", f"visible_gt{threshold:g}_rotor_std"))

    orders = np.arange(1, n_harmonics + 1, dtype=np.float64)
    for label, lo, hi in (("low", 2, 8), ("mid", 9, 24), ("high", 25, n_harmonics)):
        keep = (orders >= lo) & (orders <= hi)
        if keep.sum() < 2:
            continue
        x = np.log10(orders[keep])
        x = x - x.mean()
        denominator = np.sum(x * x)
        slopes = (
            np.einsum(
                "k,crk->cr", x, margin[:, :, keep] - margin[:, :, keep].mean(axis=2, keepdims=True)
            )
            / denominator
        )
        columns.extend((slopes.mean(axis=1), slopes.std(axis=1)))
        names.extend((f"slope_{label}_mean", f"slope_{label}_rotor_std"))

    blade_reference = margin[:, :, [0, 2, 3]].mean(axis=2)
    bpf = margin[:, :, 1] - blade_reference
    columns.extend((bpf.mean(axis=1), bpf.std(axis=1)))
    names.extend(("bpf_prominence_mean", "bpf_prominence_rotor_std"))
    low_top = min(8, n_harmonics)
    rotor_prominence = np.median(margin[:, :, :low_top], axis=2).std(axis=1)
    columns.append(rotor_prominence)
    names.append("low_order_rotor_prominence")
    features = np.column_stack(columns).reshape(n_clips, -1)
    return features, names


def _auc_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    real = np.flatnonzero(labels == 0)
    synth = np.flatnonzero(labels == 1)
    aucs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        indices = np.concatenate(
            (rng.choice(real, real.size, replace=True), rng.choice(synth, synth.size, replace=True))
        )
        auc = roc_auc_score(labels[indices], scores[indices])
        aucs[i] = max(float(auc), 1.0 - float(auc))
    auc = float(roc_auc_score(labels, scores))
    auc = max(auc, 1.0 - auc)
    lo, hi = np.quantile(aucs, (0.025, 0.975))
    return auc, float(lo), float(hi)


def two_sample_classifier(
    train_real: np.ndarray,
    test_real: np.ndarray,
    train_synthetic: np.ndarray,
    test_synthetic: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    """Train on one recording and estimate held-out separability on another."""
    train_x = np.concatenate((train_real, train_synthetic))
    train_y = np.concatenate(
        (
            np.zeros(train_real.shape[0], dtype=int),
            np.ones(train_synthetic.shape[0], dtype=int),
        )
    )
    test_x = np.concatenate((test_real, test_synthetic))
    test_y = np.concatenate(
        (
            np.zeros(test_real.shape[0], dtype=int),
            np.ones(test_synthetic.shape[0], dtype=int),
        )
    )
    classifier = make_pipeline(
        RobustScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        ),
    )
    classifier.fit(train_x, train_y)
    scores = classifier.predict_proba(test_x)[:, 1]
    auc, auc_lo, auc_hi = _auc_interval(test_y, scores, seed=seed, n_bootstrap=n_bootstrap)
    return {
        "classifier_auc": auc,
        "classifier_auc_95": [auc_lo, auc_hi],
    }


def latent_topology_diagnostic(
    real_train: ProfileDraws,
    real_test: ProfileDraws,
    synthetic_train: ProfileDraws,
    synthetic_test: ProfileDraws,
    *,
    seed: int = 0,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Diagnostic on fitted latent profiles; this is not a waveform realism gate."""
    train_real_x, names = topology_features(real_train)
    test_real_x, test_names = topology_features(real_test)
    train_synth_x, synth_names = topology_features(synthetic_train)
    test_synth_x, synth_test_names = topology_features(synthetic_test)
    if not (names == test_names == synth_names == synth_test_names):
        raise ValueError("all topology draws must share one order grid")
    classifier = two_sample_classifier(
        train_real_x,
        test_real_x,
        train_synth_x,
        test_synth_x,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )

    real_margin = real_test.peak_margin_db
    synth_margin = synthetic_test.peak_margin_db
    k = min(real_margin.shape[2], synth_margin.shape[2])
    lo, hi = np.quantile(synth_margin[:, :, :k], (0.05, 0.95), axis=(0, 1))
    coverage = float(np.mean((real_margin[:, :, :k] >= lo) & (real_margin[:, :, :k] <= hi)))
    real_curve = np.median(real_margin[:, :, :k], axis=(0, 1))
    synth_curve = np.median(synth_margin[:, :, :k], axis=(0, 1))
    scale = np.subtract(*np.quantile(real_margin[:, :, :k], (0.75, 0.25), axis=(0, 1)))
    standardized_curve_rmse = float(
        np.sqrt(np.mean(((synth_curve - real_curve) / np.maximum(scale, 1.0)) ** 2))
    )
    consistent = bool(
        classifier["classifier_auc_95"][1] < 0.70
        and coverage >= 0.80
        and standardized_curve_rmse <= 1.0
    )
    return {
        "scope": "fitted-latent-diagnostic",
        "realism_gate": False,
        "latent_consistent": consistent,
        **classifier,
        "predictive_90_coverage": coverage,
        "median_curve_iqr_rmse": standardized_curve_rmse,
        "feature_names": names,
    }


__all__ = [
    "ORDER_POINTS",
    "ProfileDraws",
    "fitted_profile_draws",
    "synthetic_profile_draws",
    "topology_features",
    "latent_topology_diagnostic",
    "two_sample_classifier",
]
