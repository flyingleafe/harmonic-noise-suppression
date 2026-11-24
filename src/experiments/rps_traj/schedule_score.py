"""Score free-running paths on an externally supplied flight schedule.

Short phases: keep the ordered joint path and three physically interpretable
readouts (peak collective speed, maximum instantaneous rotor spread, RMS
0.2 s increment). Settled cruise: compare the distribution of the existing
mean-speed + complex cross-spectral feature, not random oscillation phase.
All normalizers are functions of independent training recordings, never the
observation or generated draws. Lower is better; no duration term.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

from experiments.rps_traj.phase_dynamics import FS, INC_S, WINDOWS
from experiments.rps_traj.scores import energy_score
from experiments.rps_traj.spectral_validation import CruiseMetric


def _short_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lag = round(FS * INC_S)
    path = x.reshape(len(x), -1) / np.sqrt(x.shape[1] * x.shape[2])
    peak = x.mean(axis=1).max(axis=-1)
    spread = np.ptp(x, axis=1).max(axis=-1)
    inc_rms = np.sqrt(np.mean((x[:, :, lag:] - x[:, :, :-lag]) ** 2, axis=(1, 2)))
    return path, np.column_stack((peak, spread, inc_rms)) / np.sqrt(3)


def _score(block: np.ndarray, n_ref: int, floor: float) -> tuple[float, float]:
    # First row: observation. Next rows: reference recordings, one per donor.
    scale = max(float(np.median(pdist(block[1 : 1 + n_ref]))), floor)
    return energy_score(block[1 + n_ref :], block[0]), scale


def score_phase(
    phase: str,
    obs: np.ndarray,
    refs: np.ndarray,
    draws: np.ndarray,
    hover: float,
    metric: CruiseMetric,
) -> dict[str, float]:
    """One held-out event/origin; score is dimensionless, raw short scores in rev/s."""
    if len(refs) < 2 or not np.isfinite(hover) or hover <= 0:
        raise ValueError("need two independent real donors and positive training hover")
    x = np.concatenate((obs[None], refs, draws))
    if not np.isfinite(x).all():
        raise ValueError("non-finite observed, reference or generated trajectory")
    floor = 0.01 * hover
    if phase == "cruise":
        z = metric.transform(x)
        raw, scale = _score(z, len(refs), floor / metric.level_scale_rps)
        return {
            "score": raw / scale,
            "joint_es": raw,
            "joint_scale": scale,
            "level_es_rps": energy_score(z[1 + len(refs) :, :4], z[0, :4]) * metric.level_scale_rps,
            "spectral_es": energy_score(z[1 + len(refs) :, 4:], z[0, 4:]) * metric.spectrum_scale,
        }
    if phase not in WINDOWS or x.shape[-1] <= round(FS * INC_S):
        raise ValueError(f"invalid short phase: {phase} {x.shape}")
    path, summaries = _short_features(x)
    path_es, path_scale = _score(path, len(refs), floor)
    summary_es, summary_scale = _score(summaries, len(refs), floor)
    return {
        "score": 0.5 * (path_es / path_scale + summary_es / summary_scale),
        "path_es_rps": path_es,
        "summary_es_rps": summary_es,
        "path_scale_rps": path_scale,
        "summary_scale_rps": summary_scale,
    }
