"""Regulated motor model in total thrust and relative motor allocation.

For identical rotors total thrust is proportional to sum(w_i**2). Its square
root is an RMS speed. The three other coordinates describe relative thrust
shares with orthogonal log contrasts; the inverse maps exactly to four
nonnegative speeds and conserves the specified total thrust. A given relative
allocation error cannot produce cruise-scale separation while motors idle.
"""

import numpy as np
from scipy.special import softmax

from experiments.rps_traj import bridge_program as bp
from experiments.rps_traj import population_bridge

LOG_SCALE = 32.0  # fixed coordinate unit (rev/s per natural-log speed ratio)


class AllocationBridge(population_bridge.PopulationBridge):
    regulated = True

    @staticmethod
    def encode(w):
        w = np.maximum(w, bp.FLOOR).T
        rms = np.sqrt(np.mean(w * w, axis=1))
        return np.column_stack(
            (
                bp._inverse(rms) - bp._inverse(bp.FLOOR),
                (np.log(w) @ bp.MIXER[:, 1:]) * (LOG_SCALE / 4.0),
            )
        )

    @staticmethod
    def decode(z):
        rms = bp.LINK * np.logaddexp(0.0, (z[..., :1] + bp._inverse(bp.FLOOR)) / bp.LINK)
        shares = softmax((2.0 / LOG_SCALE) * (z[..., 1:] @ bp.MIXER[:, 1:].T), axis=-1)
        w = 2 * rms * np.sqrt(shares)
        return np.where(w <= bp.FLOOR, 0.0, w).swapaxes(-2, -1)

    def to_dict(self):
        return super().to_dict() | {"family": "allocation_bridge", "log_scale": LOG_SCALE}


def fit(tracks, *, measured=None, max_iter=80):
    m = population_bridge.fit(
        tracks, measured=measured, max_iter=max_iter, bounded=True, model_type=AllocationBridge
    )
    # On the floored tachometer data the mapping is one-to-one above the
    # floor. Its Jacobian is constant *in the fitted parameters*, not in time.
    jac = 0.0
    for (w, _), ok in zip(
        tracks, measured if measured is not None else [True] * len(tracks), strict=True
    ):
        if not ok:
            continue
        w = np.maximum(bp.FLOOR, w)
        rms = np.sqrt(np.mean(w * w, axis=0))
        jac += float(
            np.sum(
                np.log(rms)
                - np.log(-np.expm1(-rms / bp.LINK))
                + 3 * np.log(LOG_SCALE)
                - np.log(16.0)
                - np.log(w).sum(axis=0)
            )
        )
    m.history = [ll + jac for ll in m.history]
    return m
