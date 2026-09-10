"""How far the comb's shaft sits from the rev/s label, measured from the fits.

Every per-clip fit with ``Spec.rps_offset`` carries a smooth per-rotor carrier
correction with a ``rps_offset_std`` prior. Its *static* part — the mean over a
clip — is the quantity a renderer needs: within a 4 s clip the correction is
dominated by one offset per rotor, and at order ``k`` that offset displaces the
line by ``k`` times as much, which is why the top of a real comb sits far lower
against its local floor than a comb rendered exactly on the label.

Two properties matter for how this is used:

* The estimate is a **posterior mean under a zero-centred prior**, so it is
  shrunk toward zero. Transferring it therefore *understates* the label error —
  the conservative direction.
* The sub-clip structure of the correction is **not** identified: its measured
  autocorrelation is the prior's own (knots every ``rps_offset_dt_s``), so only
  the static part is reported and only the static part is rendered.

Only training recordings may be read here. The caller passes the fit directory
and the clip-id prefixes it owns; a prefix that matches nothing raises.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CarrierError:
    """Static shaft-minus-label offset of one rig, in rev/s."""

    static_scale_rps: float
    static_scale_q05_q95: list[float]
    static_std_rps: float
    within_clip_std_rps: float
    per_clip_rotor_offsets: list[list[float]]
    clips: list[str]
    variant: str
    n_bootstrap: int


#: Gaussian-consistency constant: sigma = MAD / 0.6745.
MAD_TO_SIGMA = 1.0 / 0.6745


def _robust_scale(values: np.ndarray) -> float:
    """MAD-based scale about zero.

    The per-rotor estimates are heavy-tailed for a measurable reason: a rotor
    whose lines are wide has a poorly localized carrier, and exactly those
    rotors carry the ±3–4 rev/s outliers (on Michael's, the rotor with the
    largest fitted ``width_scale``). A plain standard deviation is then a
    measurement of the estimator's failure mode rather than of the rig's label
    error, so the scale is taken robustly, about zero because the prior and the
    quantity itself are zero-centred.
    """
    return float(np.median(np.abs(values)) * MAD_TO_SIGMA)


def _clip_offsets(path: Path) -> np.ndarray:
    """``(R, N)`` fitted carrier correction of one clip, in rev/s."""
    with np.load(path, allow_pickle=True) as handle:
        params = json.loads(str(handle["params"]))
    offsets = np.asarray(params.get("rps_offset", []), dtype=np.float64)
    if offsets.ndim != 2 or offsets.size == 0:
        raise ValueError(f"{path}: fit carries no rps_offset — refit with Spec.rps_offset")
    return offsets


def fit_carrier_error(
    fit_dir: str | Path,
    *,
    clip_prefixes: tuple[str, ...],
    variant: str,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> CarrierError:
    """Measure the static carrier offset over one rig's training clips."""
    root = Path(fit_dir) / variant
    paths = sorted(
        path
        for path in root.glob("*.npz")
        if any(path.stem.startswith(prefix) for prefix in clip_prefixes)
    )
    if not paths:
        raise ValueError(f"{root}: no fit matches {clip_prefixes}")
    offsets = [_clip_offsets(path) for path in paths]
    static = np.stack([row.mean(axis=1) for row in offsets])  # (clips, rotors)
    within = np.concatenate([(row - row.mean(axis=1, keepdims=True)).ravel() for row in offsets])
    rng = np.random.default_rng(seed)
    # Resample CLIPS, not (clip, rotor) cells: one clip's rotors share its
    # telemetry and its flight state.
    draws = [
        _robust_scale(static[rng.integers(0, static.shape[0], size=static.shape[0])])
        for _ in range(n_bootstrap)
    ]
    return CarrierError(
        static_scale_rps=_robust_scale(static),
        static_scale_q05_q95=[float(value) for value in np.quantile(draws, (0.05, 0.95))],
        static_std_rps=float(static.std()),
        within_clip_std_rps=float(within.std()),
        per_clip_rotor_offsets=[[float(value) for value in row] for row in static],
        clips=[path.stem for path in paths],
        variant=variant,
        n_bootstrap=int(n_bootstrap),
    )


def apply_carrier_error(summary: dict[str, Any], carrier: CarrierError) -> dict[str, Any]:
    """Attach the measured label error to a candidate summary."""
    out = copy.deepcopy(summary)
    out["carrier_error"] = asdict(carrier)
    return out


__all__ = ["CarrierError", "apply_carrier_error", "fit_carrier_error"]
