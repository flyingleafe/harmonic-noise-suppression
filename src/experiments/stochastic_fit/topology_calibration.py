"""Gaussian synthetic-likelihood calibration of the rendered order profile.

The Whittle fit supplies floor, widths, dynamics, microphone structure and a
first profile population. Its registered raw-waveform check can still expose a
systematic renderer-space profile error. This module estimates only that error
from the training recording: a Gaussian likelihood for the difference of real
and matched-render median line/floor margins, with a second-difference Gaussian
prior. The held-out recording is never used by the calibration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class TopologyCalibration:
    reference_order: int
    line_floor_shift_db: float
    profile_correction_db: np.ndarray
    profile_basis_db: np.ndarray
    selected_rank: int
    rank_cv_nll_per_order: list[float]
    rank_cv_se_per_order: list[float]
    smoothing_lambda: float
    observed_orders: int
    standardized_rmse_before: float
    standardized_rmse_after: float

    def export(self) -> dict[str, Any]:
        out = asdict(self)
        out["profile_correction_db"] = self.profile_correction_db.tolist()
        out["profile_basis_db"] = self.profile_basis_db.tolist()
        return out


def _order_location_and_variance(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 3:
        raise ValueError("topology margins must have shape (clips, rotors, orders)")
    n_orders = values.shape[2]
    location = np.full(n_orders, np.nan)
    variance = np.full(n_orders, np.nan)
    count = np.zeros(n_orders, dtype=int)
    for order in range(n_orders):
        cell = values[:, :, order]
        cell = cell[np.isfinite(cell)]
        count[order] = cell.size
        if cell.size:
            location[order] = float(np.median(cell))
            mad = float(np.median(np.abs(cell - location[order])))
            robust_std = max(1.4826 * mad, 1.0)
            variance[order] = robust_std**2 / cell.size
    return location, variance, count


def fit_topology_calibration(
    real_train: np.ndarray,
    synthetic_train: np.ndarray,
    *,
    reference_order: int = 2,
    lambda_grid: np.ndarray | None = None,
    max_rank: int = 3,
) -> TopologyCalibration:
    """Fit the renderer-space mean correction with training-clip cross-validation."""
    real = np.asarray(real_train, dtype=np.float64)
    synthetic = np.asarray(synthetic_train, dtype=np.float64)
    if real.shape != synthetic.shape:
        if real.shape[1:] != synthetic.shape[1:] or synthetic.shape[0] % real.shape[0] != 0:
            raise ValueError(
                "synthetic topology must have an integer number of matched draws per real clip"
            )
        draws = synthetic.shape[0] // real.shape[0]
        synthetic = np.nanmedian(
            synthetic.reshape(real.shape[0], draws, *real.shape[1:]),
            axis=1,
        )
    difference_samples = real - synthetic
    difference, variance, count = _order_location_and_variance(difference_samples)
    observed = np.isfinite(difference) & (count >= 3)
    if observed.sum() < 4:
        raise ValueError("fewer than four orders have enough paired observations")
    reference_index = int(reference_order) - 1
    if reference_index < 0 or reference_index >= difference.size or not observed[reference_index]:
        valid = np.flatnonzero(observed)
        reference_index = int(valid[np.argmin(np.abs(valid - reference_index))])
    n_orders = difference.size
    second_difference = np.zeros((max(n_orders - 2, 0), n_orders), dtype=np.float64)
    for row in range(n_orders - 2):
        second_difference[row, row : row + 3] = (1.0, -2.0, 1.0)
    roughness = second_difference.T @ second_difference
    anchor = np.zeros((n_orders, n_orders), dtype=np.float64)
    anchor[reference_index, reference_index] = 1e6

    def solve(
        location: np.ndarray,
        location_variance: np.ndarray,
        location_count: np.ndarray,
        smoothing: float,
    ) -> tuple[float, np.ndarray]:
        usable = np.isfinite(location) & (location_count >= 3)
        weight = np.where(usable, 1.0 / np.maximum(location_variance, 0.25), 0.0)
        level = float(location[reference_index])
        target = np.where(usable, location - level, 0.0)
        precision = (
            np.diag(weight) + float(smoothing) * roughness + anchor + 1e-8 * np.eye(n_orders)
        )
        correction = np.linalg.solve(precision, weight * target)
        correction -= correction[reference_index]
        return level, correction

    lambdas = (
        np.logspace(-3.0, 3.0, 25)
        if lambda_grid is None
        else np.asarray(lambda_grid, dtype=np.float64)
    )
    n_clips = difference_samples.shape[0]
    if n_clips < 3:
        raise ValueError("topology calibration needs at least three matched clips")
    n_folds = min(5, n_clips)
    fold_index = np.arange(n_clips) % n_folds
    best_score = float("inf")
    smoothing = float(lambdas[0])
    for candidate in lambdas:
        errors: list[np.ndarray] = []
        for fold in range(n_folds):
            train = difference_samples[fold_index != fold]
            validation = difference_samples[fold_index == fold]
            location, fold_variance, fold_count = _order_location_and_variance(train)
            if not np.isfinite(location[reference_index]):
                continue
            level, correction = solve(location, fold_variance, fold_count, float(candidate))
            prediction = level + correction
            residual = validation - prediction[None, None, :]
            valid = np.isfinite(residual)
            robust_std = np.sqrt(np.maximum(fold_variance * np.maximum(fold_count, 1), 1.0))
            errors.append(
                (residual[valid] / np.broadcast_to(robust_std, residual.shape)[valid]) ** 2
            )
        if not errors:
            continue
        score = float(np.mean(np.concatenate(errors)))
        if score < best_score:
            best_score = score
            smoothing = float(candidate)

    def complete(values: np.ndarray) -> np.ndarray:
        reduced = np.nanmedian(values, axis=1)
        column = np.nanmedian(reduced, axis=0)
        column = np.where(np.isfinite(column), column, 0.0)
        return np.where(np.isfinite(reduced), reduced, column[None, :])

    real_complete = complete(real)
    synthetic_complete = complete(synthetic)
    rank_values = tuple(range(max(0, int(max_rank)) + 1))
    fold_scores = np.full((n_folds, len(rank_values)), np.inf, dtype=np.float64)
    fold_index = np.arange(n_clips) % n_folds
    for fold in range(n_folds):
        train = fold_index != fold
        validation = fold_index == fold
        real_model = LedoitWolf().fit(real_complete[train])
        synthetic_model = LedoitWolf().fit(synthetic_complete[train])
        covariance_delta = real_model.covariance_ - synthetic_model.covariance_
        eigenvalue, eigenvector = np.linalg.eigh(0.5 * (covariance_delta + covariance_delta.T))
        order = np.argsort(eigenvalue)[::-1]
        eigenvalue = np.maximum(eigenvalue[order], 0.0)
        eigenvector = eigenvector[:, order]
        centered_validation = real_complete[validation] - real_model.location_[None, :]
        for rank in rank_values:
            covariance = synthetic_model.covariance_.copy()
            if rank:
                covariance += (eigenvector[:, :rank] * eigenvalue[:rank][None, :]) @ eigenvector[
                    :, :rank
                ].T
            covariance += 1e-5 * np.eye(n_orders)
            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0:
                continue
            solved = np.linalg.solve(covariance, centered_validation.T).T
            fold_scores[fold, rank] = float(
                0.5 * np.mean(logdet + np.sum(centered_validation * solved, axis=1)) / n_orders
            )
    rank_scores = np.mean(fold_scores, axis=0)
    rank_se = np.std(fold_scores, axis=0, ddof=1) / np.sqrt(n_folds)
    best_rank = int(np.argmin(rank_scores))
    # Small clip sets make adjacent residual ranks statistically
    # indistinguishable. The one-standard-error rule chooses the simplest
    # model supported by training folds instead of spending every available
    # covariance degree of freedom.
    selected_rank = int(
        np.flatnonzero(rank_scores <= rank_scores[best_rank] + rank_se[best_rank])[0]
    )
    real_model = LedoitWolf().fit(real_complete)
    synthetic_model = LedoitWolf().fit(synthetic_complete)
    covariance_delta = real_model.covariance_ - synthetic_model.covariance_
    eigenvalue, eigenvector = np.linalg.eigh(0.5 * (covariance_delta + covariance_delta.T))
    order = np.argsort(eigenvalue)[::-1]
    eigenvalue = np.maximum(eigenvalue[order], 0.0)
    eigenvector = eigenvector[:, order]
    profile_basis = (
        (np.sqrt(eigenvalue[:selected_rank])[:, None] * eigenvector[:, :selected_rank].T)
        if selected_rank
        else np.zeros((0, n_orders), dtype=np.float64)
    )
    level_shift, correction = solve(difference, variance, count, smoothing)
    scale = np.sqrt(np.maximum(variance[observed], 0.25))
    before = float(np.sqrt(np.mean((difference[observed] / scale) ** 2)))
    after_error = difference[observed] - level_shift - correction[observed]
    after = float(np.sqrt(np.mean((after_error / scale) ** 2)))
    return TopologyCalibration(
        reference_order=reference_index + 1,
        line_floor_shift_db=level_shift,
        profile_correction_db=correction,
        profile_basis_db=profile_basis,
        selected_rank=selected_rank,
        rank_cv_nll_per_order=rank_scores.tolist(),
        rank_cv_se_per_order=rank_se.tolist(),
        smoothing_lambda=smoothing,
        observed_orders=int(observed.sum()),
        standardized_rmse_before=before,
        standardized_rmse_after=after,
    )


def apply_topology_calibration(
    summary: dict[str, Any], calibration: TopologyCalibration
) -> dict[str, Any]:
    """Return a JSON-safe summary whose population renders the calibrated mean."""
    import copy

    out = copy.deepcopy(summary)
    profile = np.asarray(out["rig"]["profile_db"], dtype=np.float64)
    correction = calibration.profile_correction_db
    if correction.size < profile.size:
        correction = np.pad(correction, (0, profile.size - correction.size), mode="edge")
    profile = profile + correction[: profile.size]
    out["rig"]["profile_db"] = profile.tolist()
    out["population"]["line_floor_mean_db"] = float(
        out["population"]["line_floor_mean_db"] + calibration.line_floor_shift_db
    )
    residual_basis = calibration.profile_basis_db
    if residual_basis.shape[1] < profile.size:
        residual_basis = np.pad(
            residual_basis,
            ((0, 0), (0, profile.size - residual_basis.shape[1])),
            mode="constant",
        )
    old_value = out["rig"].get("profile_basis_db")
    old_basis = (
        np.zeros((0, profile.size), dtype=np.float64)
        if old_value is None
        else np.asarray(old_value, dtype=np.float64)
    )
    basis = np.concatenate(
        (old_basis[:, : profile.size], residual_basis[:, : profile.size]),
        axis=0,
    )
    out["rig"]["profile_basis_db"] = basis.tolist()
    out["topology_calibration"] = calibration.export()
    return out


__all__ = [
    "TopologyCalibration",
    "apply_topology_calibration",
    "fit_topology_calibration",
]
