"""Where a refined rotor-speed label is allowed to replace the telemetry.

The F_VK/L-BFGS refiner maximises a comb fitness over the whole recording. That
is the right objective while the rotors are flying and the comb is strong, and
the wrong one while they idle: a standby shaft is nearly static, its comb is
weak and dense, and any correction the refiner finds there is fitted to noise.
Measured on the committed sidecars: DREGON ``free-flight_nosource_room1`` is
corrected by -0.8 to -2.0 rev/s throughout standby, and FLY125 shows a
+1.1 rev/s excursion on one rotor exactly at the ramp.

So refinement is **gated by regime**, and the gate is a policy, not an estimate:

* **standby** (slowest rotor below ``standby_max_rps``): the label is the
  telemetry, exactly.
* **cruise** (slowest rotor at or above ``cruise_min_rps``, and settled): the
  label is the refined trajectory, exactly.
* **ramp** (in between): the correction is blended, so the label is continuous
  and no step is introduced at either boundary.

The blend acts on the CORRECTION, not on the trajectories:

    r(t) = r_telemetry(t) + w(t) * (r_refined(t) - r_telemetry(t))

which makes the two end points exact by construction and keeps the result
continuous whenever ``w`` is continuous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Below this (slowest rotor, rev/s) the rig is standing by and the telemetry
#: is used unchanged. FLY125 idles at 30-40 rev/s and cruises above 65; DREGON
#: idles near 15 and cruises near 70-85.
STANDBY_MAX_RPS = 45.0
#: At or above this the rig is cruising and the refined label is used in full.
CRUISE_MIN_RPS = 65.0
#: How long the rig must have been in cruise before the refined label is
#: trusted at full weight. This is what damps the transient the refiner leaves
#: at the end of a ramp, where the comb is still sweeping fast.
SETTLE_S = 1.0


@dataclass(frozen=True)
class GatePolicy:
    standby_max_rps: float = STANDBY_MAX_RPS
    cruise_min_rps: float = CRUISE_MIN_RPS
    settle_s: float = SETTLE_S

    def as_dict(self) -> dict[str, float]:
        return dict(
            standby_max_rps=self.standby_max_rps,
            cruise_min_rps=self.cruise_min_rps,
            settle_s=self.settle_s,
        )


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """``3x^2 - 2x^3`` on [0, 1], clamped. Zero slope at both ends."""
    u = np.clip(x, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def regime_weight(
    ft: np.ndarray, r_telemetry: np.ndarray, policy: GatePolicy = GatePolicy()
) -> np.ndarray:
    """``(N,)`` weight on the refined correction, in [0, 1].

    The regime is a property of the whole rig, so it is read from the SLOWEST
    rotor: a window is standby if any rotor is still idling. The weight is a
    smoothstep in that rate, so it is continuous in time without any temporal
    filtering (the rate itself is continuous), and it is exactly 0 at the
    standby threshold and exactly 1 at the cruise threshold.

    ``settle_s`` then holds the weight down for the first ``settle_s`` seconds
    after the rig leaves standby, ramping it in linearly, which is where the
    refiner's ramp transient sits. It is keyed on standby exit rather than on
    cruise entry so that both factors vanish at the same instant and the
    envelope stays continuous and monotone across the merge.
    """
    ft = np.asarray(ft, dtype=np.float64)
    lo = np.nanmin(np.asarray(r_telemetry, dtype=np.float64), axis=0)
    span = max(policy.cruise_min_rps - policy.standby_max_rps, 1e-9)
    w = _smoothstep((lo - policy.standby_max_rps) / span)

    if policy.settle_s > 0.0 and ft.size > 1:
        # Delay full trust for settle_s after the rig LEAVES STANDBY, not after
        # it reaches cruise. Keyed on cruise entry the envelope is
        # discontinuous: just below the cruise threshold the rate term is
        # already ~1, and at the first cruising sample the elapsed time is 0, so
        # the minimum would drop the weight to 0 and ramp it back up - a step in
        # the middle of the merge, which is exactly what the merge exists to
        # avoid. Keyed on standby exit both terms are 0 at the same instant, so
        # the envelope is continuous and monotone through the ramp.
        out_of_standby = lo >= policy.standby_max_rps
        since = np.zeros_like(ft)
        # A recording that BEGINS out of standby has no ramp to settle after:
        # treat it as already settled, or the first settle_s seconds of every
        # mid-flight recording (and of any clip shorter than settle_s) would be
        # silently un-refined.
        entry = ft[0] - policy.settle_s if out_of_standby[0] else np.nan
        for i in range(ft.size):
            if out_of_standby[i]:
                if not np.isfinite(entry):
                    entry = ft[i]
                since[i] = ft[i] - entry
            else:
                entry = np.nan
                since[i] = 0.0
        w = np.minimum(w, np.clip(since / policy.settle_s, 0.0, 1.0))
    return w


def gate_refinement(
    ft: np.ndarray,
    r_telemetry: np.ndarray,
    r_refined: np.ndarray,
    policy: GatePolicy = GatePolicy(),
) -> tuple[np.ndarray, np.ndarray]:
    """``(labels, weight)`` with refinement applied only where it is trusted.

    ``labels`` equals ``r_telemetry`` wherever the rig is standing by and
    ``r_refined`` wherever it is cruising and settled, with the correction
    blended in between.
    """
    tel = np.asarray(r_telemetry, dtype=np.float64)
    ref = np.asarray(r_refined, dtype=np.float64)
    if tel.shape != ref.shape:
        raise ValueError(f"shape mismatch: telemetry {tel.shape} vs refined {ref.shape}")
    w = regime_weight(ft, tel, policy)
    return tel + w[None, :] * (ref - tel), w


def regime_summary(
    ft: np.ndarray, r_telemetry: np.ndarray, policy: GatePolicy = GatePolicy()
) -> dict[str, float]:
    """Fraction of frames in each regime, for the sidecar report."""
    lo = np.nanmin(np.asarray(r_telemetry, dtype=np.float64), axis=0)
    w = regime_weight(ft, r_telemetry, policy)
    n = max(lo.size, 1)
    return dict(
        frames=float(lo.size),
        standby_frac=float(np.mean(lo < policy.standby_max_rps)),
        ramp_frac=float(np.mean((lo >= policy.standby_max_rps) & (lo < policy.cruise_min_rps))),
        cruise_frac=float(np.mean(lo >= policy.cruise_min_rps)),
        weight_zero_frac=float(np.sum(w <= 0.0) / n),
        weight_one_frac=float(np.sum(w >= 1.0) / n),
    )
