"""Render from a control policy, fit the render, compare with what was drawn.

A policy is a claim about a distribution, and a claim nobody checked is a bug
waiting for a training run to find. This closes the loop: the renderer draws a
parameter set and produces audio, the fitting machinery sees only the audio and
the speed labels, and the two parameter sets are compared.

It tests both directions at once, which is why it is worth the compute:

* the STREAM is what it says it is — C1's fitted line widths must collapse to
  the analysis floor and its fitted amplitude drift to zero, because a static
  comb has neither; C2's must come back inside the ranges the policy sampled;
* the FIT can read its own generative model — if a parameter cannot be
  recovered from a render of the model that contains it exactly, no result
  about real data resting on that parameter means anything.

WHAT IS COMPARED, and why not everything. The fit normalizes the periodogram to
unit mean, so absolute level is unidentifiable by construction and the profile
is compared after removing a common offset. Only orders the clip can actually
support are compared: ``identify.marginal_information`` decides which, so an
order whose exported value is the prior's is not counted as a fit failure.

The rendered width in ``line_mode: fm`` is not the fit's ``gamma`` parameter —
the renderer builds a tone bank whose lines are broadened by a shaft that
wanders (Gaussian, half width ``1.177 k sigma``) and a per-harmonic phase
diffusion (Lorentzian, ``q k``). The comparison is therefore against the
effective half width those two imply, which is what a spectrum shows and what
the fit's ``gamma`` is trying to be.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import Clip, Periodogram, periodogram

#: HWHM of a Gaussian whose standard deviation is 1.
_GAUSS_HWHM = float(np.sqrt(2.0 * np.log(2.0)))


@dataclass
class Rendered:
    """One control clip and the parameters that produced it."""

    clip: Clip
    pg: Periodogram
    params: Any  # StochasticParams
    diag: dict[str, Any]


def render_from_policy(
    policy_path: str,
    *,
    arm: int = 0,
    n_clips: int = 3,
    seed: int = 0,
    duration_s: float = 4.0,
    n_mics: int | None = 2,
    sample_rate: int = 16000,
) -> list[Rendered]:
    """Render clips from one noise arm of a policy, keeping the draws.

    ``n_mics`` overrides the arm's microphone count: the fit cost is linear in
    it and nothing being checked here is per-microphone, so two is enough.
    """
    import yaml

    from data_processing.stochastic_rotor_noise import StochasticNoisePool

    policy = yaml.safe_load(Path(policy_path).read_text())
    arms = [a for a in policy["sources"]["noise"] if a.get("kind") == "stochastic"]
    if arm >= len(arms):
        raise ValueError(f"policy has {len(arms)} stochastic arms, asked for index {arm}")
    cfg = dict(arms[arm])
    if n_mics is not None:
        cfg["n_mics"] = int(n_mics)
        for key in ("fixed_mic_gain_db", "fixed_mic_floor_db", "fixed_mic_gain_all_db"):
            if isinstance(cfg.get("ranges"), dict):
                cfg["ranges"].pop(key, None)
    pool = StochasticNoisePool.from_config(
        cfg, duration_s=duration_s, sample_rate=int(policy.get("sample_rate", sample_rate))
    )
    rng = np.random.default_rng(seed)
    out: list[Rendered] = []
    for _ in range(n_clips):
        audio, rps, params, diag = pool.render(rng, duration_s)
        clip = Clip(
            clip_id=f"roundtrip_{len(out):02d}",
            group="synthetic",
            audio=np.asarray(audio, dtype=np.float32),
            rps=np.asarray(rps, dtype=np.float32),
            sr=pool.sample_rate,
        )
        out.append(Rendered(clip=clip, pg=periodogram(clip), params=params, diag=diag))
    return out


def effective_gamma_hz(params: Any, k: np.ndarray) -> np.ndarray:
    """``(R,K)`` half width the renderer's tone bank actually produces.

    The shaft wander is coherent across a rotor's orders, so it smears order
    ``k`` by ``k sigma`` rev/s of standard deviation; the per-harmonic phase
    diffusion adds a Lorentzian ``q k``. A Voigt profile's width is close to
    the sum of the two half widths, which is all the precision this comparison
    needs.
    """
    jitter = np.atleast_1d(np.asarray(params.shaft_jitter_rps, dtype=np.float64))
    if jitter.size == 1:
        jitter = np.repeat(jitter, params.n_rotors)
    q = float(getattr(params, "phase_diffusion_hz_per_order", 0.0) or 0.0)
    return _GAUSS_HWHM * jitter[:, None] * k[None, :] + q * k[None, :]


def compare(rendered: Rendered, fitted: dict[str, Any], *, max_std_db: float = 3.0) -> dict:
    """Planted against fitted, per clip."""
    from .identify import marginal_information

    planted = rendered.params
    params = fitted["params"]
    prof_fit = np.asarray(params["profile_db"], dtype=np.float64)
    prof_true = np.asarray(planted.profile_db, dtype=np.float64)
    n_rotors = min(prof_fit.shape[0], prof_true.shape[0])
    n_harm = min(prof_fit.shape[1], prof_true.shape[1])
    prof_fit, prof_true = prof_fit[:n_rotors, :n_harm], prof_true[:n_rotors, :n_harm]
    k = np.arange(1, n_harm + 1, dtype=np.float64)

    info = marginal_information(
        params,
        freqs=rendered.pg.freqs,
        times=rendered.pg.times,
        rps=rendered.pg.rps,
        n_mics=rendered.clip.audio.shape[0],
        clip_id=rendered.clip.clip_id,
        line_shape="gauss",
    )
    ok = info.identifiable(max_std_db)[:n_rotors, :n_harm]
    # The fit normalizes the clip's power, so only the SHAPE is comparable.
    offset = float(np.median(prof_fit[ok] - prof_true[ok])) if ok.any() else 0.0
    err = (prof_fit - offset) - prof_true

    gamma_fit = np.asarray(params["gamma"], dtype=np.float64)[:n_rotors, :n_harm]
    gamma_true = effective_gamma_hz(planted, k)[:n_rotors]
    h_db = np.asarray(fitted["params"].get("h_db", np.zeros((n_rotors, n_harm, 1))))
    floor_level = np.asarray(params["floor_level_db"], dtype=np.float64)

    def at(order: int, arr: np.ndarray) -> float:
        return float(np.median(arr[:, min(order, arr.shape[1]) - 1]))

    return {
        "clip": rendered.clip.clip_id,
        "identifiable_fraction": float(ok.mean()),
        "per_rotor_identifiable": [int(v) for v in ok.sum(axis=1)],
        "profile_rms_error_db": float(np.sqrt(np.mean(err[ok] ** 2))) if ok.any() else float("nan"),
        "profile_correlation": (
            float(np.corrcoef(prof_fit[ok], prof_true[ok])[0, 1]) if ok.sum() > 2 else float("nan")
        ),
        "profile_level_offset_db": offset,
        "gamma_hz": {
            f"k={order}": {"planted": at(order, gamma_true), "fitted": at(order, gamma_fit)}
            for order in (2, 8, 32)
        },
        "amp_drift_std_db": {
            "planted": float(planted.harm_gp_std_db),
            "fitted": float(np.std(h_db)),
        },
        "floor_drift_std_db": {
            "planted": float(planted.floor_gp_std_db),
            "fitted": float(np.std(floor_level)),
        },
        "floor_tilt_db_oct": {
            "planted": float(planted.floor_tilt_db_oct),
            "fitted": float(params["floor_tilt_db_oct"]),
        },
        "shaft_jitter_rps_planted": [
            float(v) for v in np.atleast_1d(np.asarray(planted.shaft_jitter_rps))
        ],
        "nll_fit": fitted["scores"]["nll_fit"],
        "nll_loo": fitted["scores"]["nll_loo"],
    }


def roundtrip(
    policy_path: str,
    *,
    arm: int = 0,
    n_clips: int = 3,
    seed: int = 0,
    duration_s: float = 4.0,
    n_mics: int = 2,
    k_cap: int = 64,
    iters: tuple[int, int, int, int] = (120, 120, 240, 40),
    log: Any = print,
) -> dict[str, Any]:
    """Render, fit and compare; returns one record per clip."""
    from .fit import fit_clip
    from .model import make_spec

    rendered = render_from_policy(
        policy_path,
        arm=arm,
        n_clips=n_clips,
        seed=seed,
        duration_s=duration_s,
        n_mics=n_mics,
    )
    records = []
    for item in rendered:
        spec = make_spec(
            item.pg,
            n_mics=item.clip.audio.shape[0],
            f_max=None,
            k_cap=k_cap,
            variant={"rps_offset": True, "line_shape": "gauss"},
        )
        log(
            f"  fitting {item.clip.clip_id}: K={spec.n_harm} "
            f"frames={item.pg.times.size} freqs={item.pg.freqs.size} "
            f"mics={item.clip.audio.shape[0]}",
            flush=True,
        )
        fitted = fit_clip(item.pg, spec, iters=iters, log=log)
        record = compare(item, fitted)
        records.append(record)
        log(
            f"  DONE {record['clip']}: profile rms {record['profile_rms_error_db']:.2f} dB "
            f"(r={record['profile_correlation']:.3f}, {record['identifiable_fraction']:.2f} "
            f"identifiable) | gamma k=8 planted "
            f"{record['gamma_hz']['k=8']['planted']:.3f} fitted "
            f"{record['gamma_hz']['k=8']['fitted']:.3f} | amp drift planted "
            f"{record['amp_drift_std_db']['planted']:.2f} fitted "
            f"{record['amp_drift_std_db']['fitted']:.2f}",
            flush=True,
        )
    return {"policy": policy_path, "arm": arm, "clips": records}
