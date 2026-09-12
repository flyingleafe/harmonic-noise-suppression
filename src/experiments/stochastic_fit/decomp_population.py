"""Michael profile population from linewidth-matched Vold--Kalman amplitudes.

Unlike the union-comb waveform statistic, these envelopes already separate
microphone, physical rotor and harmonic order against the exact carrier used by
the solve.  FLY125's middle time block selects factor rank; FLY124 is not read
by this fitter.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import tdseries as td
from sklearn.decomposition import FactorAnalysis


@dataclass(frozen=True)
class DecompProfileModel:
    mean_relative_db: np.ndarray
    rotor_delta_db: np.ndarray
    profile_basis_db: np.ndarray
    residual_std_db: np.ndarray
    selected_rank: int
    rank_valid_loglik_per_order: list[float]
    rank_valid_se_per_order: list[float]
    n_train_chunks: int
    n_valid_chunks: int
    chunk_s: float

    def export(self) -> dict[str, Any]:
        out = asdict(self)
        for name in (
            "mean_relative_db",
            "rotor_delta_db",
            "profile_basis_db",
            "residual_std_db",
        ):
            out[name] = getattr(self, name).tolist()
        return out


def _fit_profile_samples(
    profiles: np.ndarray,
    relative_time: np.ndarray,
    *,
    max_rank: int,
    valid_fraction: float,
) -> DecompProfileModel:
    """Fit ``mu + delta_r + Bz + epsilon`` to chunk×rotor dB profiles."""
    if profiles.ndim != 3:
        raise ValueError("profiles must have shape (chunk, rotor, order)")
    if relative_time.shape != (profiles.shape[0],):
        raise ValueError("relative_time must have one value per chunk")
    half = 0.5 * float(valid_fraction)
    valid_chunk = np.abs(relative_time - 0.5) < half
    train_chunk = ~valid_chunk
    if train_chunk.sum() < 4 or valid_chunk.sum() < 1:
        raise ValueError("decomposition profile split has too few chunks")
    train = profiles[train_chunk]
    valid = profiles[valid_chunk]
    mean = np.median(train, axis=(0, 1))
    rotor_delta = np.median(train - mean[None, None, :], axis=0)
    rotor_delta -= rotor_delta.mean(axis=0, keepdims=True)

    def residual(values: np.ndarray) -> np.ndarray:
        return (values - mean[None, None, :] - rotor_delta[None]).reshape(-1, profiles.shape[2])

    x = residual(train)
    validation = residual(valid)
    max_rank = min(int(max_rank), x.shape[0] - 1, x.shape[1])
    variance = np.var(x, axis=0) + 1e-3
    diagonal_samples = (
        -0.5
        * (np.log(2.0 * np.pi * variance)[None, :] + np.square(validation) / variance[None, :]).sum(
            axis=1
        )
        / x.shape[1]
    )
    score_samples: list[np.ndarray] = [diagonal_samples]
    models: list[FactorAnalysis | None] = [None]
    for rank in range(1, max_rank + 1):
        model = FactorAnalysis(
            n_components=rank,
            random_state=0,
            max_iter=3000,
            tol=1e-4,
        ).fit(x)
        score_samples.append(model.score_samples(validation) / x.shape[1])
        models.append(model)
    # Rotors from one time chunk share the flight state; average them before
    # estimating uncertainty. The one-standard-error rule then prevents tiny,
    # noisy held-out gains from consuming every allowed factor.
    scores = [
        float(sample.reshape(valid.shape[:2]).mean(axis=1).mean()) for sample in score_samples
    ]
    score_se = [
        float(sample.reshape(valid.shape[:2]).mean(axis=1).std(ddof=1) / np.sqrt(valid.shape[0]))
        if valid.shape[0] > 1
        else 0.0
        for sample in score_samples
    ]
    best_rank = int(np.argmax(scores))
    selected_rank = next(
        rank
        for rank, score in enumerate(scores)
        if score >= scores[best_rank] - score_se[best_rank]
    )
    selected = models[selected_rank]
    if selected is None:
        basis = np.zeros((0, x.shape[1]), dtype=np.float64)
        residual_std = np.sqrt(variance)
    else:
        mean = mean + selected.mean_
        basis = np.asarray(selected.components_, dtype=np.float64)
        residual_std = np.sqrt(np.maximum(selected.noise_variance_, 1e-6))
    return DecompProfileModel(
        mean_relative_db=mean,
        rotor_delta_db=rotor_delta,
        profile_basis_db=basis,
        residual_std_db=residual_std,
        selected_rank=selected_rank,
        rank_valid_loglik_per_order=scores,
        rank_valid_se_per_order=score_se,
        n_train_chunks=int(train_chunk.sum()),
        n_valid_chunks=int(valid_chunk.sum()),
        chunk_s=float("nan"),
    )


def fit_decomp_profile(
    frame: td.Frame,
    *,
    k_max: int = 64,
    chunk_s: float = 2.0,
    min_rps: float = 30.0,
    valid_fraction: float = 0.1,
    max_rank: int = 8,
) -> DecompProfileModel:
    """Fit the profile hierarchy from one decomposed training recording."""
    amp = np.asarray(frame["amp"].data, dtype=np.float64)
    amp_valid = np.asarray(frame["amp_valid"].data, dtype=bool)
    rps = np.asarray(frame["rps"].data, dtype=np.float64)
    env_stride = int(round(rps.shape[-1] / amp.shape[-1]))
    sample_rate = int(round(float(frame["rps"].tindex.rate)))
    env_rate = sample_rate / env_stride
    chunk = int(round(chunk_s * env_rate))
    k = min(int(k_max), amp.shape[2])
    rows: list[np.ndarray] = []
    relative_time: list[float] = []
    for start in range(0, amp.shape[-1] - chunk + 1, chunk):
        sample_start = start * env_stride
        sample_stop = (start + chunk) * env_stride
        if float(rps[:, sample_start:sample_stop].mean()) < min_rps:
            continue
        value = amp[:, :, :k, start : start + chunk]
        valid = amp_valid[:, :k, start : start + chunk]
        denominator = np.maximum(valid.sum(axis=-1)[None, :, :], 1)
        power = np.sum(np.square(value) * valid[None], axis=-1) / denominator
        profile = 10.0 * np.log10(np.maximum(power.mean(axis=0), 1e-20))
        reference = min(1, k - 1)
        profile -= profile[:, reference : reference + 1]
        rows.append(profile)
        relative_time.append((start + 0.5 * chunk) / amp.shape[-1])
    if not rows:
        raise ValueError("decomposition profile has no eligible complete chunks")
    fitted = _fit_profile_samples(
        np.stack(rows),
        np.asarray(relative_time),
        max_rank=max_rank,
        valid_fraction=valid_fraction,
    )
    return DecompProfileModel(**{**asdict(fitted), "chunk_s": float(chunk_s)})


def apply_decomp_profile(summary: dict[str, Any], model: DecompProfileModel) -> dict[str, Any]:
    """Replace only the static profile population; keep Whittle floor/dynamics."""
    out = copy.deepcopy(summary)
    old_mean = np.asarray(out["rig"]["profile_db"], dtype=np.float64)
    n_harmonics = old_mean.size
    k = model.mean_relative_db.size
    anchor = old_mean[min(1, n_harmonics - 1)]
    mean = model.mean_relative_db - model.mean_relative_db[min(1, k - 1)] + anchor
    if k < n_harmonics:
        tail_order = np.arange(k + 1, n_harmonics + 1, dtype=np.float64)
        slope = (mean[-1] - mean[-8]) / np.log10(k / (k - 7)) if k >= 8 else 0.0
        tail = mean[-1] + slope * np.log10(tail_order / k)
        mean = np.r_[mean, tail]
    delta = model.rotor_delta_db
    basis = model.profile_basis_db
    residual = model.residual_std_db
    if k < n_harmonics:
        delta = np.pad(delta, ((0, 0), (0, n_harmonics - k)), mode="edge")
        basis = np.pad(basis, ((0, 0), (0, n_harmonics - k)), mode="constant")
        residual = np.pad(residual, (0, n_harmonics - k), mode="edge")
    out["rig"]["profile_db"] = mean[:n_harmonics].tolist()
    out["rig"]["delta_db"] = delta[:, :n_harmonics].tolist()
    out["rig"]["profile_basis_db"] = basis[:, :n_harmonics].tolist()
    out["decomp_profile_model"] = model.export()
    out["profile_residual_std_db"] = residual[:n_harmonics].tolist()
    out.pop("visibility_model", None)
    return out


__all__ = [
    "DecompProfileModel",
    "apply_decomp_profile",
    "fit_decomp_profile",
]
