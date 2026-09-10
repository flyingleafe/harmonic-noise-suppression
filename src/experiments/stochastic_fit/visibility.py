"""Training-only hurdle model for harmonic-order visibility.

The continuous rig population remains the visible component.  A Bernoulli
order state selects a second, attenuated Gaussian component rather than setting
a tooth to exact zero.  Probabilities and attenuation are fitted only from the
registered raw-waveform line/local-floor margins on the training recording.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class VisibilityModel:
    threshold_db: float
    visible_probability: np.ndarray
    attenuation_mean_db: np.ndarray
    attenuation_std_db: float
    smoothing_lambda: float
    cv_nll_per_observation: list[float]

    def export(self) -> dict[str, Any]:
        out = asdict(self)
        out["visible_probability"] = self.visible_probability.tolist()
        out["attenuation_mean_db"] = self.attenuation_mean_db.tolist()
        return out


def _fit_logits(
    visible: np.ndarray,
    observed: np.ndarray,
    smoothing: float,
) -> np.ndarray:
    """Penalized binomial logits with smooth order trend, parity and BPF terms."""
    n_orders = visible.shape[1]
    order = np.arange(1, n_orders + 1, dtype=np.float64)
    even = (order % 2 == 0).astype(np.float64)
    bpf = (order == 2).astype(np.float64)
    second = np.zeros((max(n_orders - 2, 0), n_orders), dtype=np.float64)
    for row in range(n_orders - 2):
        second[row, row : row + 3] = (1.0, -2.0, 1.0)
    successes = np.where(observed, visible, 0.0).sum(axis=0)
    trials = observed.sum(axis=0)
    empirical = (successes + 1.0) / (trials + 2.0)
    ridge = 0.05
    initial = np.r_[np.log(empirical / (1.0 - empirical)), 0.0, 0.0]

    def objective(parameter: np.ndarray) -> tuple[float, np.ndarray]:
        trend = parameter[:n_orders]
        eta = trend + parameter[-2] * even + parameter[-1] * bpf
        probability = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
        nll = np.sum(trials * np.logaddexp(0.0, eta) - successes * eta)
        curvature = second @ trend
        penalty = 0.5 * smoothing * np.sum(curvature**2)
        penalty += 0.5 * ridge * np.sum(parameter**2)
        residual = trials * probability - successes
        gradient = np.zeros_like(parameter)
        gradient[:n_orders] = residual + smoothing * (second.T @ curvature)
        gradient[-2] = np.sum(residual * even)
        gradient[-1] = np.sum(residual * bpf)
        gradient += ridge * parameter
        return float(nll + penalty), gradient

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 3000, "maxls": 50, "ftol": 1e-9, "gtol": 1e-6},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"visibility logistic fit failed: {result.message}")
    trend = result.x[:n_orders]
    eta = trend + result.x[-2] * even + result.x[-1] * bpf
    return np.clip(1.0 / (1.0 + np.exp(-eta)), 0.01, 0.99)


def _binomial_nll(
    probability: np.ndarray,
    visible: np.ndarray,
    observed: np.ndarray,
) -> float:
    p = np.clip(probability, 1e-8, 1.0 - 1e-8)
    cell = -(visible * np.log(p)[None, :] + (1.0 - visible) * np.log1p(-p)[None, :])
    return float(cell[observed].mean())


def _smooth_attenuation(
    raw: np.ndarray,
    weight: np.ndarray,
    smoothing: float,
) -> np.ndarray:
    n_orders = raw.size
    second = np.zeros((max(n_orders - 2, 0), n_orders), dtype=np.float64)
    for row in range(n_orders - 2):
        second[row, row : row + 3] = (1.0, -2.0, 1.0)
    precision = np.diag(weight) + smoothing * (second.T @ second) + 1e-4 * np.eye(n_orders)
    value = np.linalg.solve(precision, weight * raw)
    return np.clip(value, 3.0, 40.0)


def fit_visibility_model(
    real_train: np.ndarray,
    synthetic_train: np.ndarray,
    *,
    threshold_db: float = 6.0,
    lambda_grid: np.ndarray | None = None,
) -> VisibilityModel:
    """Fit PV from FLY125 margins, selecting smoothness by clip-level CV."""
    real = np.nanmedian(np.asarray(real_train, dtype=np.float64), axis=1)
    synthetic = np.nanmedian(np.asarray(synthetic_train, dtype=np.float64), axis=1)
    if real.ndim != 2 or synthetic.ndim != 2 or real.shape[1] != synthetic.shape[1]:
        raise ValueError("visibility margins must be (clips, rotors, common orders)")
    observed = np.isfinite(real)
    visible = np.where(observed, real > threshold_db, False)
    if observed.sum() == 0:
        raise ValueError("visibility fit has no observed real margins")
    lambdas = (
        np.logspace(-2.0, 3.0, 16)
        if lambda_grid is None
        else np.asarray(lambda_grid, dtype=np.float64)
    )
    n_clips = real.shape[0]
    n_folds = min(5, n_clips)
    fold = np.arange(n_clips) % n_folds
    scores = np.zeros(lambdas.size, dtype=np.float64)
    for index, smoothing in enumerate(lambdas):
        for heldout in range(n_folds):
            train = fold != heldout
            validation = ~train
            probability = _fit_logits(visible[train], observed[train], float(smoothing))
            scores[index] += _binomial_nll(
                probability,
                visible[validation],
                observed[validation],
            )
    scores /= n_folds
    best = int(np.argmin(scores))
    smoothing = float(lambdas[best])
    # The off component is genuinely left-censored: attenuate from the
    # continuous component's 90th percentile, not its median, so a high-factor
    # draw does not remain visible after selecting the off state.
    synthetic_median = np.nanquantile(synthetic, 0.9, axis=0)

    target = np.full(real.shape[1], threshold_db - 3.0, dtype=np.float64)
    faint_count = np.zeros(real.shape[1], dtype=np.float64)
    faint_residuals: list[np.ndarray] = []
    for order in range(real.shape[1]):
        faint = real[:, order]
        faint = faint[np.isfinite(faint) & (faint <= threshold_db)]
        faint_count[order] = faint.size
        if faint.size:
            target[order] = float(np.median(faint))
            faint_residuals.append(faint - target[order])
    raw_attenuation = np.maximum(synthetic_median - target, 3.0)
    attenuation = _smooth_attenuation(
        raw_attenuation,
        np.maximum(faint_count, 1.0),
        max(smoothing, 0.1),
    )
    residual = np.concatenate(faint_residuals) if faint_residuals else np.array([3.0])
    mad = float(np.median(np.abs(residual - np.median(residual))))
    attenuation_std = float(np.clip(1.4826 * mad, 1.0, 6.0))
    return VisibilityModel(
        threshold_db=float(threshold_db),
        visible_probability=probability,
        attenuation_mean_db=attenuation,
        attenuation_std_db=attenuation_std,
        smoothing_lambda=smoothing,
        cv_nll_per_observation=scores.tolist(),
    )


def apply_visibility_model(summary: dict[str, Any], model: VisibilityModel) -> dict[str, Any]:
    out = copy.deepcopy(summary)
    n_harmonics = len(out["rig"]["profile_db"])
    probability = model.visible_probability
    attenuation = model.attenuation_mean_db
    if probability.size < n_harmonics:
        probability = np.pad(probability, (0, n_harmonics - probability.size), mode="edge")
        attenuation = np.pad(attenuation, (0, n_harmonics - attenuation.size), mode="edge")
    exported = model.export()
    exported["visible_probability"] = probability[:n_harmonics].tolist()
    exported["attenuation_mean_db"] = attenuation[:n_harmonics].tolist()
    out["visibility_model"] = exported
    return out


__all__ = ["VisibilityModel", "apply_visibility_model", "fit_visibility_model"]
