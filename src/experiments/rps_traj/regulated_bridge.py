"""Learn ground-regulation poles from the same complete-flight likelihood.

Two extra parameters: closed-loop memory in collective speed, and in the three
contrast axes. All ground phases use the same fit, no separately fixed idle
level. The disturbance is propagated across arming, takeoff and touchdown.
"""

import numpy as np

from experiments.rps_traj import population_bridge


class RegulatedBridge(population_bridge.PopulationBridge):
    regulated = True

    def to_dict(self):
        return super().to_dict() | {"family": "regulated_bridge"}

    def coefficients(self):
        """Flat deployable law: independent values only, not optimizer history."""
        p = self.p
        free = np.ones(population_bridge.BINS, bool)
        free[[0, population_bridge.ARMED_BIN]] = False
        parts = [
            *p["beta"],
            p["rho"][:-1],
            p["rho"][-1, :2],
            p["basevar"],
            p["activity"][1:],
            p["boost"],
            p["pulse"],
            p["trans"],
            p["mix"][:-1],
            self.q[free],
            self.support,
        ]
        values = np.concatenate([np.ravel(a) for a in parts])
        if len(values) != self.n_params:
            raise ValueError("parameter count does not match the portable law")
        return values

    @classmethod
    def from_coefficients(cls, values):
        """Decode a physical-tachometry fit without a recording or training pack."""
        v = np.asarray(values, dtype=np.float64)
        # The only varying size is the number of persistent styles: 4x3
        # level/shelf coefficients, 4 activity scales, 3 hazards and a mixture
        # probability per added style.
        if v.ndim != 1 or not np.isfinite(v).all() or len(v) not in (236, 256, 276):
            raise ValueError("invalid regulated-bridge coefficient vector")
        k = 1 + (len(v) - 236) // 20
        pos = 0

        def take(*shape):
            nonlocal pos
            n = int(np.prod(shape))
            x = v[pos : pos + n].reshape(shape)
            pos += n
            return x

        beta = [
            take(sum(len(g) - 2 for g in population_bridge.MEAN_GRIDS[int(h != 0)]) + 3 * k)
            for h in range(4)
        ]
        rho = np.vstack((take(7, 4), take(2)[[0, 1, 1, 1]]))
        basevar = take(7, 4)
        activity = np.vstack((np.ones((1, 4)), take(k - 1, 4)))
        boost, pulse, trans, masses = take(4), take(4), take(k, 3), take(k - 1)
        mix = np.r_[masses, 1.0 - masses.sum()]
        q = np.ones((population_bridge.BINS, 2))
        free = np.ones(len(q), bool)
        free[[0, population_bridge.ARMED_BIN]] = False
        q[free] = take(int(free.sum()), 2)
        support = float(take(1)[0])
        if (
            pos != len(v)
            or (mix < 0).any()
            or (q <= 0).any()
            or (basevar <= 0).any()
            or support <= 0.32
        ):
            raise ValueError("invalid regulated-bridge law")
        p = dict(
            beta=beta,
            rho=rho,
            basevar=basevar,
            activity=activity,
            phasevar=np.ones((6, 4)),
            boost=boost,
            pulse=pulse,
            trans=trans,
            mix=mix,
        )
        return cls(p, np.ones(k, bool), [], q, support)


def fit(tracks, *, measured=None, max_iter=80):
    return population_bridge.fit(
        tracks, measured=measured, max_iter=max_iter, bounded=True, model_type=RegulatedBridge
    )
