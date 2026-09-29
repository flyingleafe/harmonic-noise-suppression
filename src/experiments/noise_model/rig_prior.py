"""Rigs drawn from the UNFITTED v3 prior a fit recorded, optionally guarded.

:func:`draw_rig` redraws every RIG site of a ``noise-v3-fit/1`` payload from
the prior serialized in it (``fit["priors"]``, so a round-4 fit's draw follows
``PRIORS_V4`` and its ``--priors-set`` overrides), at the recording's own
measured quantities (``fit["diagnostics"]["measured"]``: the profile centre,
the floor mean and ``sigma_B``, the wind level) and its shaft rate ``lam``.
Only the fitted numbers are replaced; everything measured is the base fit's.

One payload is one REGIME's prior: the recorded priors are marginal per fit,
and nothing couples a standby draw to a cruise draw, so a two-regime rig is not
drawn here.

:func:`sample_rigs` draws until ``n`` rigs pass an acceptance callable
(:func:`gate_acceptance`: the identifiability gate of :mod:`.identifiability`
on a frame pool) and raises :class:`SamplingExhausted` when the attempts run out.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from . import identifiability as ID

__all__ = [
    "GuardedDraw",
    "SamplingExhausted",
    "Verdict",
    "draw_rig",
    "gate_acceptance",
    "sample_rigs",
]

#: Redraws of a negative ``amp_exp`` before :func:`draw_rig` gives up.
MAX_SPEED_LAW_REDRAWS = 1000


def draw_rig(fit: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """One draw of every RIG site from the v3 prior the fit recorded.

    The laws are the model's (``model._sample_params_v3``): ``sigma_nu`` from
    its LogNormal or HalfNormal, ``gamma_rk ~ HalfNormal(floor_k + c gamma_0 k)``
    (the floor by blade-pass parity), ``profile_db ~ N(measured centre, sd)``,
    ``z ~ N(0, I)`` on the floor spline at the measured ``mu``/``sigma_B``, the
    speed laws at their pins or priors, the wind at ``N(measured, sd)``. The
    wander hyperparameters are the fit's (measured, fixed). A speed law must
    rise with speed: a negative ``amp_exp`` draw is REJECTED and redrawn
    (``floor_exp`` is LogNormal, positive by construction).
    """
    pr, p = fit["priors"], fit["params"]
    meas = fit["diagnostics"]["measured"]
    pins = fit["diagnostics"]["span_pins"]
    prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
    r_, k_ = prof.shape
    ks = np.arange(1, k_ + 1, dtype=np.float64)
    g0, gc = float(pr["gamma_hz"]["gamma0_hz"]), float(pr["gamma_hz"]["gamma_c"])
    bpf, other = (float(v) for v in pr["gamma_hz"].get("gamma_floor_hz") or (0.0, 0.0))
    blades = int(pr["gamma_hz"].get("blades") or 2)
    gamma_floor = np.where(ks % blades == 0, bpf, other)
    gamma = np.abs(rng.normal(0.0, np.broadcast_to(gamma_floor + gc * g0 * ks, (r_, k_))))
    if pr["sigma_nu"]["family"] == "LogNormal":
        sigma_nu = float(np.exp(rng.normal(pr["sigma_nu"]["log_mu"], pr["sigma_nu"]["log_sd"])))
    else:
        sigma_nu = abs(float(rng.normal(0.0, float(pr["sigma_nu"]["scale_rad_s"]))))
    centre = np.asarray(meas["profile_centre_db"], dtype=np.float64)[:r_, :k_]
    profile = rng.normal(centre, float(pr["profile_db"]["sd"]))
    pinned = set(pins.get("pinned") or [])

    def law(name: str, draw: Any) -> float:
        return float(pins["values"][name]) if name in pinned else float(draw())

    def positive_amp_exp() -> float:
        for _ in range(MAX_SPEED_LAW_REDRAWS):
            a = float(rng.normal(*pr["amp_exp"]))
            if a > 0.0:
                return a
        raise ValueError(f"amp_exp prior {pr['amp_exp']} gave no positive draw")

    amp_exp = law("amp_exp", positive_amp_exp)
    floor_exp = law("floor_exp", lambda: np.exp(rng.normal(*pr["log_floor_exp"])))
    static = law("floor_static_rel", lambda: np.exp(rng.normal(*pr["log_floor_static"])))
    z = rng.normal(size=len(p["floor"]["floor_shape_z"]))
    out = copy.deepcopy(fit)
    q = out["params"]
    q["sigma_nu"] = sigma_nu
    q["gamma_hz"] = gamma.tolist()
    q["profile"]["profile_db"] = profile.tolist()
    q["profile"]["amp_exp"] = amp_exp
    q["floor"]["floor_shape_z"] = z.tolist()
    q["floor"]["floor_exp"] = floor_exp
    q["floor"]["floor_static_rel"] = static
    if q.get("wind") is not None:
        q["wind"]["wind_db"] = rng.normal(
            np.asarray(meas["wind_db"], dtype=np.float64), float(pr["wind"]["sd"])
        ).tolist()
    return out


class Verdict(Protocol):
    """What an acceptance callable returns (:class:`identifiability.GateResult` is one)."""

    passed: bool
    reason: str


@dataclass
class GuardedDraw:
    """One draw and its verdict: ``attempt`` is the index of its RNG stream."""

    fit: dict[str, Any]
    result: Verdict | None
    attempt: int


class SamplingExhausted(RuntimeError):
    """Fewer than ``n`` draws were accepted within the attempt budget."""

    def __init__(self, n: int, accepted: list[GuardedDraw], rejected: list[GuardedDraw]):
        self.accepted, self.rejected = accepted, rejected
        reasons = Counter(d.result.reason.split(" (")[0] for d in rejected if d.result is not None)
        top = "; ".join(f"{k} x{v}" for k, v in reasons.most_common(4))
        super().__init__(
            f"{len(accepted)} of {n} rigs accepted in {len(accepted) + len(rejected)} draws; "
            f"rejections: {top}"
        )


def gate_acceptance(
    pool: ID.FramePool, config: ID.GateConfig | None = None
) -> Callable[[dict[str, Any]], ID.GateResult]:
    """The identifiability gate on ``pool`` as an acceptance callable. A pool
    that cannot test some rotor (:func:`identifiability.pool_coverage`) raises
    here: no draw could pass it."""
    cov = ID.pool_coverage(pool, config)
    if cov.untestable is not None:
        raise ValueError(f"the frame pool cannot test this gate: {cov.untestable}")
    return lambda fit: ID.gate(fit, pool, config)


def sample_rigs(
    fit: dict[str, Any],
    *,
    n: int,
    seed: int,
    accept: Callable[[dict[str, Any]], Verdict] | None = None,
    max_attempts: int = 200,
) -> tuple[list[GuardedDraw], list[GuardedDraw]]:
    """``(accepted, rejected)``: prior draws of ``fit`` until ``n`` pass
    ``accept`` (``None``: every draw). Attempt ``i`` draws from
    ``default_rng([seed, i])``, so a draw is reproducible on its own. Fewer
    than ``n`` within ``max_attempts`` raises :class:`SamplingExhausted`."""
    accepted: list[GuardedDraw] = []
    rejected: list[GuardedDraw] = []
    for i in range(int(max_attempts)):
        if len(accepted) >= int(n):
            break
        draw = draw_rig(fit, np.random.default_rng([int(seed), i]))
        verdict = None if accept is None else accept(draw)
        (accepted if verdict is None or verdict.passed else rejected).append(
            GuardedDraw(draw, verdict, i)
        )
    if len(accepted) < int(n):
        raise SamplingExhausted(int(n), accepted, rejected)
    return accepted, rejected
