"""Every generation of this project's rotor-noise model, on ONE trajectory.

This module merges the two noise notebooks it replaces — ``noise_v2_sampler``
(the fitted v2 rigs) and ``stochastic_noise_lab`` (the hand-written stochastic
family, with sliders) — into one place where a clip of EVERY generation can be
rendered on the SAME rotor-speed track, under the SAME level rule, with the
SAME render seed, and then compared by ear, by spectrogram and by number.

THE FOUR GENERATIONS, as noise sources::

    LegacyRandom(seed=0)          a fresh draw of the hand-written family
                                  (data_processing.stochastic_rotor_noise),
                                  which is what `kind: stochastic` renders
                                  when a policy gives it `ranges:`
    LegacyBank("easy", 0)         entry 0 of data/rig_banks/rig_easy_n2048.json
                                  — the exact preset bank the `rig_easy` /
                                  `rig_hard` arms trained on
    V2Fit("dregon")               the round-5 UNCALIBRATED single-regime fit;
                                  V2Fit("michaels") is the round-3 standby +
                                  cruise pair with the 45-65 rev/s smoothstep
    V2Bank("easy", 0)             entry 0 of data/rig_banks/noise_v2_easy_
                                  n2048.json (= `dload:noise-v2-banks`), the
                                  sampled v2 rig bank

WHY ONE TRAJECTORY.  The training policies these generations come from differ
in four things at once — the level rule, render/flight reuse, the speed scaling
(``rps_scale_range``) and the trajectory source — so two arms' clips are never
comparable as heard.  Here the trajectory is drawn ONCE
(:func:`trajectory`), every source renders on it (:func:`render_all`), and one
level rule is applied to all of them, so what is left between two clips is the
NOISE MODEL and nothing else.

    traj    = trajectory("fitted", rig="dregon", seed=0, duration_s=10.0)
    sources = [LegacyRandom(seed=0), LegacyBank("easy", 0), V2Fit("dregon"), ...]
    frames  = render_all(sources, traj, seed=0, level=("window", 0.1))
    show(frames)                      # the spectrogram grid + rps
    players(frames)                   # one audio widget per generation
    line_stats(frames)                # the R4 order prominence, as a table
    tune(sources[0], traj)            # the sliders, on one source

THE SLIDERS.  ``stochastic_noise_lab.Lab`` drove an ipywidgets panel from
inside the module, and :func:`tune` keeps it: the same amplitude means, wander
amplitudes and times, harmonic coherence, floor-colour wander and clip length,
the same "New random parameters" / "Regenerate" buttons and status line, the
same semantics (a slider move changes THAT NUMBER and leaves the draw's static
random parts alone).  What it drops is what the merge made shared — the
rotor-speed source, the aggressiveness, the speed scale and the level mode,
which now come from the trajectory and from ``level=`` — so a panel clip stays
comparable with the clips of :func:`render_all`.  The fitted generations are
not tunable and their panel says so: a bank entry's numbers are shown
read-only, and a v2 payload exposes ``comb_offset_db`` alone.  Everything else
is a plain function API over which the notebook is thin cells (the project's
convention: logic in the ``.py``, the ``.ipynb`` a driver).

THE RATE CONTRACT (inherited from ``noise_v2_sampler``).  The v2 renderer wants
``rps_rev_s`` as ``(R, T)`` on its OWN OUTPUT GRID — one carrier sample per
audio sample at 16 kHz — and the legacy ``synthesize`` wants the same.  The
trajectory fits are stated on a 100 Hz grid and their whole content lives below
20 Hz, so a trajectory is drawn on a slow grid and linearly interpolated to
16 kHz.  Every Frame here therefore carries TWO rotor tracks: ``rps`` decimated
to 100 Hz (what the plots draw) and ``rps_render``, the exact 16 kHz carrier a
renderer consumes.

THE LEVEL RULE.  ``level=None`` keeps each generation's NATIVE scale: absolute
fitted units for the v2 sources (the fit's own rendered RMS), and the legacy
model's arbitrary internal scale for the legacy sources — which is exactly why
the notebook's default is ``("window", 0.1)``, a per-clip RMS that makes the
generations audible against each other.  ``("flight", rms)`` is the training
policy's other rule: the number is the level AT THE REFERENCE SPEED and the
clip's own floor envelope (``mean_r(speed**floor_exp) + floor_static_rel``, the
factor each renderer shaped its floor with) is put back, so a slow passage stays
quieter than a fast one.  Each source reports its own envelope, so the rule
means the same thing across generations.

MEMORY.  Rendering only: no forward model is ever evaluated on a fit POOL.
:func:`expected_vs_realised` evaluates the v2 forward model on a few seconds of
ONE clip, which is the largest thing here.
"""

from __future__ import annotations

import copy
import json
import subprocess
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td

ROOT = Path(__file__).resolve().parent.parent

#: The renderers' output rate, and therefore the rate ``rps_render`` is on.
SR = 16000

#: The grid the fitted trajectory model is stated on (``params.RATE_HZ``).
TRAJ_FS = 100.0

#: Rate the plotted ``rps`` track is carried at.
RPS_PLOT_SR = 100

#: Grid the hand-written scaffold trajectories are generated on before
#: interpolation (:mod:`data_processing.rps_synthesis` runs a per-sample filter
#: in Python, so generating it at the audio rate wastes seconds).
RPS_FS = 400.0

#: The trajectory fits bundle (``dload pull rps-traj-fits``).
TRAJ_FITS = "dload:rps-traj-fits"

#: The v2 noise fits of each rig, by regime.  ``"single"`` is a one-regime rig.
FIT_PATHS: dict[str, dict[str, str]] = {
    "dregon": {
        "single": "results/noise_v2/rounds/round5/fits/dregon_room2_floor__flight_profile.json",
    },
    "michaels": {
        "standby": "results/noise_v2/rounds/round3/fits/michaels_fly125_standby__flight.json",
        "cruise": "results/noise_v2/rounds/round3/fits/michaels_fly125_cruise__flight.json",
    },
}

#: Trajectory-model rig name per v2 noise rig.
TRAJ_RIG = {"dregon": "dregon", "michaels": "michaels"}

RIGS = tuple(FIT_PATHS)

#: The LEGACY preset banks the ``rig_easy`` / ``rig_hard`` arms trained on
#: (``conf/online_mix/rig_{easy,hard}_5050.yaml``).  Gitignored build products;
#: :func:`data_processing.stochastic_rotor_noise.load_preset_bank` says how to
#: rebuild one when it is missing.
LEGACY_BANKS = {
    "easy": "data/rig_banks/rig_easy_n2048.json",
    "hard": "data/rig_banks/rig_hard_n2048.json",
}

#: The v2 rig banks (the same files as ``dload:noise-v2-banks``).
V2_BANKS = {
    "easy": "data/rig_banks/noise_v2_easy_n2048.json",
    "hard": "data/rig_banks/noise_v2_hard_n2048.json",
}

#: ``line_mode`` each legacy generation is rendered with: the bank arms declare
#: ``line_mode: fm`` in their policies, while a bare ``ranges:`` draw is the
#: family's own default.  Both are constructor arguments; these are the
#: defaults, and :meth:`NoiseSource.describe` prints which one is in force.
LEGACY_RANDOM_LINE_MODE = "stochastic"
LEGACY_BANK_LINE_MODE = "fm"

#: The deleted lab's slider table, lifted field for field: ``{parameter:
#: (min, max, step, label)}``.  The first two are the amplitude means and the
#: rest are the covariance parameters of the Gaussian processes, plus the share
#: of the broadband floor that does NOT follow the rotors.  :func:`tune` builds
#: its panel from this; nothing else reads it.
LEGACY_SLIDERS: dict[str, tuple[float, float, float, str]] = {
    "harm_mean_db": (-20.0, 20.0, 0.5, "harmonic level (dB)"),
    "floor_mean_db": (-40.0, 10.0, 0.5, "broadband level (dB)"),
    "harm_gp_std_db": (0.0, 12.0, 0.25, "harmonic wander (dB)"),
    "harm_gp_tau_s": (0.1, 12.0, 0.1, "harmonic wander time (s)"),
    "harm_coherence": (0.0, 1.0, 0.05, "harmonic coherence"),
    "floor_gp_std_db": (0.0, 10.0, 0.25, "floor wander (dB)"),
    "floor_gp_tau_s": (0.2, 20.0, 0.2, "floor wander time (s)"),
    "floor_tilt_gp_std": (0.0, 3.0, 0.05, "floor color wander (dB/oct)"),
    "floor_tilt_gp_tau_s": (1.0, 30.0, 0.5, "floor color time (s)"),
    "floor_static_rel": (0.0, 0.30, 0.005, "recording floor (rel)"),
}

#: ``line_mode`` values :func:`tune` offers a legacy source.  The old lab
#: offered the first two; ``fm`` is what the bank arms declare.
LEGACY_LINE_MODES = ("stochastic", "coherent", "fm")

#: The comb-offset range :func:`tune` gives a v2 source, in dB.
COMB_OFFSET_RANGE = (-12.0, 12.0, 0.25)

#: Published frames datasets a ``"real"`` trajectory can be pulled from.
DATASETS = ("DREGON-frames", "michaels-frames")

#: Trajectory kinds :func:`trajectory` understands.
TRAJ_KINDS = ("fitted", "real", "full_flight", "intermittent", "ou")


def _repo_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - no git
        return "unknown"


def _pool_span(fit: dict[str, Any]) -> tuple[float, float] | None:
    """The carrier range the fit's OWN objective saw, as it recorded it.

    ``diagnostics.batch.carrier_{min,max}_rev_s``.  Outside it the fitted speed
    envelopes (``profile.amp_exp``, ``floor.floor_exp``) EXTRAPOLATE, which is
    the one thing worth watching when a sampled trajectory drives the render.
    """
    batch = ((fit.get("diagnostics") or {}).get("batch") or {}) if fit else {}
    lo, hi = batch.get("carrier_min_rev_s"), batch.get("carrier_max_rev_s")
    if lo is None or hi is None:
        return None
    return (float(lo), float(hi))


def _offset_fit(fit: dict[str, Any], comb_offset_db: float) -> dict[str, Any]:
    """A DEEP COPY of one v2 fit with ``comb_offset_db`` on every rotor's
    ``profile_db``.  The loaded payloads are never mutated."""
    if not comb_offset_db:
        return fit
    copied = copy.deepcopy(fit)
    prof = np.asarray(copied["params"]["profile"]["profile_db"], dtype=np.float64)
    copied["params"]["profile"]["profile_db"] = (prof + float(comb_offset_db)).tolist()
    return copied


def _floor_gain(rps: np.ndarray, *, ref: float, exp: float, static_rel: float) -> float:
    """``sqrt(mean_t(mean_r((rps/ref)**exp) + static_rel))`` — the AMPLITUDE the
    ``"flight"`` level rule puts back.

    Both renderers shape their floor with this exact factor as a POWER (hence
    the square root): :func:`data_processing.stochastic_rotor_noise.build_psd`'s
    ``floor_gain`` and :meth:`data_processing.noise_v2_pool.NoiseV2Pool.
    _apply_level`'s envelope are the same expression with each model's own
    reference speed, exponent and static share.
    """
    speed = np.maximum(np.asarray(rps, dtype=np.float64), 0.0) / max(float(ref), 1e-6)
    envelope = (speed ** float(exp)).mean(axis=0) + max(float(static_rel), 0.0)
    return float(np.sqrt(max(float(np.mean(envelope)), 0.0)))


def _restrict_mics(params: Any, n_mics: int) -> Any:
    """A legacy parameter set listened to through its FIRST ``n_mics`` mics.

    A bank entry pins the per-microphone gains of the array it was fitted
    against (eight, for both rigs), and ``synthesize`` refuses a mismatched
    ``n_mics`` rather than guessing.  Taking the leading rows is the only
    reading that keeps the entry's own numbers: it is the same rig heard on
    fewer of its own microphones.  A parameter set with no pinned gains (a
    fresh ``sample_params`` draw) is returned untouched.
    """
    fixed = {
        name: getattr(params, name)
        for name in ("fixed_mic_gain_db", "fixed_mic_floor_db", "fixed_mic_gain_all_db")
        if getattr(params, name) is not None
    }
    if not fixed or all(np.shape(v)[0] == n_mics for v in fixed.values()):
        return params
    return params.with_(
        **{name: np.asarray(value, dtype=np.float64)[:n_mics] for name, value in fixed.items()}
    )


# ── the v2 rig ──────────────────────────────────────────────────────────────


class Rig:
    """One v2 rig: its fit(s), how to render them, and where they came from."""

    def __init__(
        self,
        *,
        name: str,
        fits: dict[str, dict[str, Any]],
        traj_rig: str,
        paths: dict[str, str],
        pool_rps: dict[str, tuple[float, float] | None],
        provenance: dict[str, dict[str, str]],
        repo_sha: str,
    ) -> None:
        self.name = name
        #: ``{"single": fit}`` or ``{"standby": fit, "cruise": fit}``.
        self.fits = fits
        #: Rig name inside the ``rps-traj-fits`` bundle.
        self.traj_rig = traj_rig
        #: ``{regime: repo-relative fit path}``.
        self.paths = paths
        #: ``{regime: the fit's own recorded carrier span (rev/s)}``.
        self.pool_rps = pool_rps
        #: ``{regime: (schema, git sha the fit was written at)}``.
        self.provenance = provenance
        #: Short SHA of this checkout, when the rig was loaded.
        self.repo_sha = repo_sha

    @property
    def per_regime(self) -> bool:
        """Is this the standby/cruise composition rather than a single fit?"""
        return "single" not in self.fits

    @property
    def span(self) -> tuple[float, float]:
        """The union of the fits' recorded carrier spans (rev/s)."""
        spans = [s for s in self.pool_rps.values() if s is not None]
        if not spans:
            return (float("nan"), float("nan"))
        return (min(s[0] for s in spans), max(s[1] for s in spans))

    def offset_fits(self, comb_offset_db: float) -> dict[str, dict[str, Any]]:
        """Deep copies of the fits with ``comb_offset_db`` on every rotor's
        ``profile_db``.  The loaded fits are NEVER mutated."""
        if not comb_offset_db:
            return self.fits
        return {regime: _offset_fit(fit, comb_offset_db) for regime, fit in self.fits.items()}

    def render_audio(
        self, rps: np.ndarray, *, seed: int, n_mics: int, comb_offset_db: float = 0.0
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """``((n_mics, T) float64, diagnostics)`` --- the rig's own renderer."""
        from experiments.noise_model import render as RD

        fits = self.offset_fits(comb_offset_db)
        if self.per_regime:
            audio, diag = RD.render_noise_regimes(
                fits, rps, sr=SR, n_mics=int(n_mics), seed=int(seed), return_diagnostics=True
            )
        else:
            audio, diag = RD.render_noise(
                fits["single"],
                rps,
                sr=SR,
                n_mics=int(n_mics),
                seed=int(seed),
                return_diagnostics=True,
            )
        return np.asarray(audio, dtype=np.float64), dict(diag)

    def expected_m(self, rps: np.ndarray, *, n_fft: int, hop: int, comb_offset_db: float = 0.0):
        """``(1, frames, bins)`` predicted periodogram of ONE mic on ``rps``."""
        from experiments.noise_model import render as RD

        fits = self.offset_fits(comb_offset_db)
        if self.per_regime:
            return RD.expected_periodogram_regimes(
                fits, rps, n_fft=n_fft, hop=hop, sr=SR, n_mics=1, frame_chunk=4
            )
        return RD.expected_periodogram(
            fits["single"], rps, n_fft=n_fft, hop=hop, sr=SR, n_mics=1, frame_chunk=4
        )


def load_rig(name: str) -> Rig:
    """Load one v2 rig's noise fit(s) with their provenance.

    ``name`` is ``"dregon"`` (the round-5 uncalibrated single-regime fit) or
    ``"michaels"`` (the round-3 standby + cruise pair).
    """
    key = str(name).lower()
    if key not in FIT_PATHS:
        raise ValueError(f"unknown rig {name!r}; known rigs are {list(FIT_PATHS)}")
    fits: dict[str, dict[str, Any]] = {}
    provenance: dict[str, dict[str, str]] = {}
    pool: dict[str, tuple[float, float] | None] = {}
    for regime, rel in FIT_PATHS[key].items():
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"{key} {regime} fit not found: {path}")
        fit = json.loads(path.read_text(encoding="utf-8"))
        fits[regime] = fit
        provenance[regime] = {
            "schema": str(fit.get("schema")),
            "git": str(fit.get("git", "unknown")),
            "support": str(fit.get("support")),
            "mode": str(fit.get("mode")),
        }
        pool[regime] = _pool_span(fit)
    return Rig(
        name=key,
        fits=fits,
        traj_rig=TRAJ_RIG[key],
        paths=dict(FIT_PATHS[key]),
        pool_rps=pool,
        provenance=provenance,
        repo_sha=_repo_sha(),
    )


def describe(rig: Rig, *, k_show: int = 8) -> None:
    """Print a v2 fit's key numbers, per regime, plus its provenance."""
    from experiments.noise_model import model as MD

    print(f"rig {rig.name}  ({'per-regime' if rig.per_regime else 'single-regime'})")
    print(f"  trajectory rig : {rig.traj_rig}   (bundle {TRAJ_FITS})")
    print(f"  checkout       : {rig.repo_sha}")
    for regime, fit in rig.fits.items():
        p = fit["params"]
        prov = rig.provenance[regime]
        span = rig.pool_rps[regime]
        prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
        gamma = np.asarray(MD.gamma_from_params(p), dtype=np.float64)
        print()
        print(f"  [{regime}] {rig.paths[regime]}")
        print(f"    schema {prov['schema']}  git {prov['git'][:12]}  support {prov['support']}")
        print(
            "    pool carrier   : "
            + (f"{span[0]:.1f}-{span[1]:.1f} rev/s" if span else "not recorded")
            + f"   k_max {prof.shape[1]}"
        )
        print(f"    sigma_nu {float(p['sigma_nu']):.4f}   lam {float(p['lam']):.4f}")
        k = min(int(k_show), prof.shape[1])
        print(f"    profile_db, k=1..{k} (dB, per rotor):")
        for r in range(prof.shape[0]):
            row = "  ".join(f"{v:7.2f}" for v in prof[r, :k])
            print(f"      rotor {r}: {row}")
        g1 = gamma[:, 0]
        g8 = gamma[:, min(7, gamma.shape[1] - 1)]
        print(
            "    gamma_hz k=1   : "
            + "  ".join(f"{v:.4f}" for v in g1)
            + "\n    gamma_hz k=8   : "
            + "  ".join(f"{v:.4f}" for v in g8)
        )
        fl = p["floor"]
        print(
            f"    floor          : mean {float(fl['floor_mean_db']):.2f} dB   "
            f"tilt {float(fl['floor_tilt_db_oct']):+.3f} dB/oct   "
            f"exp {float(fl['floor_exp']):.3f}   static_rel {float(fl['floor_static_rel']):.4g}"
        )
        print(f"    amp_exp        : {float(p['profile']['amp_exp']):.4f}")


# ── the trajectory ──────────────────────────────────────────────────────────


def traj_rig_names() -> tuple[str, ...]:
    """Every trajectory the bundle offers: the seven fitted rigs plus
    ``"posterior"``, the reserved name that draws a FRESH drone from the global
    rig hyperprior (``posterior.json``) for every flight."""
    from data_processing.trajectory_model.source import load_bundle

    return tuple(load_bundle(TRAJ_FITS).names)


def span_report(span: tuple[float, float] | None, rps: np.ndarray) -> dict[str, float]:
    """How far a carrier track leaves the span a fit was identified on."""
    rps = np.asarray(rps, dtype=np.float64)
    lo, hi = (float("nan"), float("nan")) if span is None else span
    return {
        "rps_min": float(rps.min()),
        "rps_max": float(rps.max()),
        "pool_min": float(lo),
        "pool_max": float(hi),
        "frac_below": float(np.mean(rps < lo)) if span is not None else 0.0,
        "frac_above": float(np.mean(rps > hi)) if span is not None else 0.0,
    }


def _fitted_rps(
    *,
    seed: int,
    duration_s: float,
    rig: str = "dregon",
    mean_shift: float = 0.0,
    mean_scale: float = 1.0,
    full_flight: bool = False,
    fs: float = TRAJ_FS,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """The FITTED trajectory model of one rig (or of a hyperprior draw).

    ``rig`` is any name in :func:`traj_rig_names` — the seven fitted rigs, or
    ``"posterior"``, the reserved name that draws a FRESH drone from the global
    rig hyperprior for the flight (its own ``mu``, trim, dynamics, level-offset
    model, idle level and ESC clamp).  ``mean_shift`` / ``mean_scale`` move the
    hover level (``mu * scale + shift`` rev/s); they are RANGES in the training
    policy and single numbers here, so one seed pins the clip.  ``full_flight``
    wraps the airborne process in ``trajectory_model.flight.wrap_airborne`` —
    ground silence, spin-up, warm-up idle, take-off, airborne, landing — which
    for ``michaels`` is what drives the carrier through the 45-65 rev/s band
    and exercises the standby/cruise blend.
    """
    from data_processing.trajectory_model.source import FittedTrajectorySource, load_bundle

    bundle = load_bundle(TRAJ_FITS)
    name = str(rig)
    if name not in bundle.names:
        raise ValueError(f"unknown trajectory rig {name!r}; bundle offers {bundle.names}")
    src = FittedTrajectorySource(
        bundle,
        {name: 1.0},
        mean_shift=(float(mean_shift), float(mean_shift)),
        mean_scale=(float(mean_scale), float(mean_scale)),
        measurement_noise=False,
    )
    rng = np.random.default_rng(int(seed))
    n_low = int(round(float(duration_s) * float(fs)))
    if full_flight:
        low = src.flight(rng, float(fs), duration_s=float(duration_s))
    else:
        draw = src.draw(rng)
        src.last_draw = draw
        low = draw.params.sample_airborne(n_low, rng, fs=float(fs), clip=draw.clip)
    drawn = src.last_draw
    meta: dict[str, Any] = {
        "traj_rig": name,
        "traj_rig_drawn": str(drawn.rig) if drawn is not None else None,
        "mean_shift": float(mean_shift),
        "mean_scale": float(mean_scale),
        "full_flight": bool(full_flight),
        "measurement_noise": False,
        "esc_clip_rev_s": list(drawn.clip) if drawn is not None else None,
        "mu_rev_s": drawn.params.mu.tolist() if drawn is not None else None,
        "hover_rev_s": float(np.mean(drawn.params.mu)) if drawn is not None else None,
        "idle_rev_s": np.asarray(drawn.idle_rps).tolist() if drawn is not None else None,
        "offset_s_c": float(drawn.params.s_c) if drawn is not None else None,
        "offset_s_r": np.asarray(drawn.params.s_r).tolist() if drawn is not None else None,
    }
    return np.atleast_2d(np.asarray(low, dtype=np.float64)), float(fs), meta


def _scaffold_rps(
    kind: str,
    *,
    seed: int,
    duration_s: float,
    aggressiveness: float = 1.0,
) -> tuple[np.ndarray, float, dict[str, Any], float]:
    """The hand-written scaffold trajectories of :mod:`data_processing.rps_synthesis`.

    ``"ou"`` is the Ornstein-Uhlenbeck model in quadrotor control-mode space
    (continuous wander), ``"intermittent"`` the pilot-and-airframe model (mostly
    steady, with occasional maneuvers, which is what a real flight looks like),
    ``"full_flight"`` a whole ground-to-ground flight from which a random window
    of ``duration_s`` is taken — so successive seeds visit the ground, the ramps
    and cruise.  Returns the low-rate track, its rate, its meta and the start
    offset of the window inside it.
    """
    from data_processing import rps_synthesis

    rng = np.random.default_rng(int(seed))
    if kind == "ou":
        low = rps_synthesis.generate(duration_s, RPS_FS, aggressiveness=aggressiveness, rng=rng)
    elif kind == "intermittent":
        low = rps_synthesis.generate_intermittent(
            duration_s, RPS_FS, aggressiveness=aggressiveness, rng=rng
        )
    else:
        low = rps_synthesis.generate_full_flight(
            None, RPS_FS, aggressiveness=aggressiveness, rng=rng
        )
    low = np.atleast_2d(np.asarray(low, dtype=np.float64))
    start = 0.0
    if kind == "full_flight":
        total = (low.shape[1] - 1) / RPS_FS
        start = float(rng.uniform(0.0, max(total - float(duration_s), 0.0)))
    meta: dict[str, Any] = {
        "traj_rig": None,
        "aggressiveness": float(aggressiveness),
        "window_start_s": start,
    }
    return low, RPS_FS, meta, start


def _real_rps(
    *,
    duration_s: float,
    dataset: str = "DREGON-frames",
    recording: str | None = None,
    offset_s: float = 10.0,
    labels: str = "telemetry",
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """A real recording's telemetry, through :func:`generator_lab.real_slice`.

    Already on the audio grid: ``real_slice`` lifts the event-sampled telemetry
    onto it, which is the project's canonical path for both rigs' rotor tracks.
    """
    from generator_lab import real_slice

    if not recording:
        raise ValueError(
            "the 'real' trajectory needs a recording id — recordings(dataset) lists them"
        )
    exc = real_slice(dataset, recording, float(offset_s), float(duration_s), labels=labels)
    rps = np.asarray(exc.rps, dtype=np.float64)
    meta: dict[str, Any] = {
        "traj_rig": None,
        "dataset": str(dataset),
        "recording": str(recording),
        "offset_s": float(offset_s),
        "labels": str(labels),
        "slice_label": str(exc.label),
    }
    return rps, float(SR), meta


def recordings(dataset: str) -> list[str]:
    """Recording ids of a published frames dataset that carry a rotor track."""
    from generator_lab import recordings as _recordings

    return _recordings(dataset)


def _traj_frame(rps: np.ndarray, meta: dict[str, Any]) -> td.Frame:
    """The two-track trajectory Frame: ``rps`` at 100 Hz, ``rps_render`` at 16 kHz."""
    step = max(int(round(SR / RPS_PLOT_SR)), 1)
    return td.Frame(
        {
            "rps": td.uniform(
                np.ascontiguousarray(rps[:, ::step]),
                RPS_PLOT_SR,
                dims=("rotor", "time"),
                t_start=0.0,
            ),
            "rps_render": td.uniform(
                np.ascontiguousarray(rps), SR, dims=("rotor", "time"), t_start=0.0
            ),
            "meta": td.Frame(meta),
        }
    )


def trajectory(
    kind: str = "fitted",
    *,
    seed: int = 0,
    duration_s: float = 10.0,
    rps_scale: float = 1.0,
    **kw: Any,
) -> td.Frame:
    """One rotor-speed trajectory, as a Frame every source can be rendered on.

    Parameters
    ----------
    kind
        ``"fitted"`` — the FITTED trajectory model (``rps-traj-fits``,
        ``measurement_noise=False`` because the audio is rendered FROM these
        labels and must not hear the ESC's sample-and-hold).  Takes ``rig=``
        (any name of :func:`traj_rig_names`, or ``"posterior"``),
        ``mean_shift=``, ``mean_scale=``, ``full_flight=``.
        ``"real"`` — a real recording's telemetry; takes ``dataset=``,
        ``recording=``, ``offset_s=``, ``labels=``.
        ``"full_flight"`` / ``"intermittent"`` / ``"ou"`` — the hand-written
        scaffold of :mod:`data_processing.rps_synthesis`; take
        ``aggressiveness=``.
    seed
        Seeds the trajectory draw only; each render has its own seed.
    rps_scale
        Multiplies the whole track, EXACTLY: a stopped rotor stays stopped and
        every other speed moves with its own comb.  This is the trajectory's
        knob alone — no source's reference speed is touched, so a scaled
        trajectory is a faster aircraft of the same design, which is what the
        training policies' ``rps_scale_range`` does to a window.

    Returns
    -------
    A Frame with ``rps`` ``(R, duration_s * 100)`` for the plots,
    ``rps_render`` ``(R, duration_s * 16000)`` for the renderers, and a ``meta``
    Frame naming the source (and, for ``"fitted"``, the DRAWN DRONE: its
    per-rotor ``mu``, hover level, warm-up idle, ESC clamp and offset stds).
    """
    key = str(kind)
    if key not in TRAJ_KINDS:
        raise ValueError(f"unknown trajectory kind {key!r}; known kinds are {list(TRAJ_KINDS)}")
    start = 0.0
    if key == "fitted":
        low, fs, meta = _fitted_rps(seed=seed, duration_s=duration_s, **kw)
    elif key == "real":
        low, fs, meta = _real_rps(duration_s=duration_s, **kw)
    else:
        low, fs, meta, start = _scaffold_rps(key, seed=seed, duration_s=duration_s, **kw)

    n_out = int(round(float(duration_s) * SR))
    t_low = np.arange(low.shape[1]) / float(fs)
    t_out = start + np.arange(n_out) / float(SR)
    rps = np.stack([np.interp(t_out, t_low, low[r]) for r in range(low.shape[0])])
    if rps_scale != 1.0:
        rps = rps * float(rps_scale)
    meta = {
        "traj_kind": key,
        "seed": int(seed),
        "duration_s": float(duration_s),
        "rps_scale": float(rps_scale),
        "traj_fs": float(fs),
        "rps_min": float(rps.min()),
        "rps_max": float(rps.max()),
        "rps_mean": float(rps.mean()),
        **meta,
    }
    return _traj_frame(rps, meta)


def describe_traj(traj: td.Frame) -> None:
    """Print WHICH trajectory a Frame carries — and, for a fitted draw, which
    drone it was flown on."""
    m = dict(traj["meta"].items())
    kind = str(m.get("traj_kind"))
    print(
        f"trajectory   : {kind}  seed {int(m['seed'])}  "
        f"{float(m['duration_s']):.1f} s  rps_scale {float(m['rps_scale']):.2f}"
    )
    print(
        f"  speeds     : {float(m['rps_min']):.1f}-{float(m['rps_max']):.1f} rev/s "
        f"(mean {float(m['rps_mean']):.1f})"
    )
    if kind == "real":
        print(f"  recording  : {m['dataset']} / {m['slice_label']}")
        return
    if kind != "fitted":
        print(f"  generator  : rps_synthesis, aggressiveness {float(m['aggressiveness']):.2f}")
        return
    drawn = m.get("traj_rig_drawn")
    asked = m.get("traj_rig")
    origin = "the rig's own fit" if asked != "posterior" else "a FRESH draw from the hyperprior"
    print(f"  fitted rig : {asked} -> {drawn}  ({origin})")
    mu = [float(v) for v in (m.get("mu_rev_s") or [])]
    print(
        f"  hover      : {float(m['hover_rev_s']):.1f} rev/s   mu "
        + " ".join(f"{v:.1f}" for v in mu)
    )
    idle = [float(v) for v in (m.get("idle_rev_s") or [])]
    clip = [float(v) for v in (m.get("esc_clip_rev_s") or [])]
    print(
        "  idle       : "
        + " ".join(f"{v:.1f}" for v in idle)
        + f"   ESC clamp {clip[0]:.1f}-{clip[1]:.1f} rev/s"
    )
    print(
        f"  offset std : common {float(m['offset_s_c']):.2f} rev/s, per-rotor "
        + " ".join(f"{float(v):.2f}" for v in (m.get("offset_s_r") or []))
    )


# ── the noise sources ───────────────────────────────────────────────────────


class NoiseSource:
    """One generation of the noise model, renderable on a given carrier.

    Subclasses carry a ``name`` (unique inside a comparison), a ``generation``
    (``"legacy"`` or ``"v2"``), an ``entry`` naming exactly which parameter set
    is being heard, and a ``span`` — the carrier range the underlying fit was
    identified on, or ``None`` for a model that has no such evidence.
    """

    name: str = ""
    generation: str = ""
    entry: str = ""
    span: tuple[float, float] | None = None

    def describe(self) -> None:
        raise NotImplementedError

    def render(
        self,
        rps_16k: np.ndarray,
        *,
        seed: int,
        n_mics: int = 1,
        comb_offset_db: float = 0.0,
    ) -> np.ndarray:
        """``(n_mics, T)`` float64 at 16 kHz, in the generation's OWN units."""
        raise NotImplementedError

    def flight_gain(self, rps_16k: np.ndarray) -> float:
        """The amplitude factor the ``("flight", rms)`` level rule puts back."""
        raise NotImplementedError


class _LegacySource(NoiseSource):
    """Shared render path of the hand-written stochastic family."""

    generation = "legacy"

    def __init__(self, params: Any, *, name: str, entry: str, line_mode: str, n_fft: int = 2048):
        self.params = params
        self.name = name
        self.entry = entry
        self.line_mode = str(line_mode)
        self.n_fft = int(n_fft)

    def render(
        self,
        rps_16k: np.ndarray,
        *,
        seed: int,
        n_mics: int = 1,
        comb_offset_db: float = 0.0,
    ) -> np.ndarray:
        from data_processing import stochastic_rotor_noise as srn

        params = _restrict_mics(self.params, int(n_mics))
        if comb_offset_db:
            # The legacy model's comb-against-floor knob is the harmonic mean
            # level; the v2 rigs' is profile_db. Same meaning, different field.
            params = params.with_(harm_mean_db=float(params.harm_mean_db) + float(comb_offset_db))
        audio, diag = srn.synthesize(
            params,
            np.asarray(rps_16k, dtype=np.float64),
            rng=np.random.default_rng(int(seed)),
            n_mics=int(n_mics),
            n_fft=self.n_fft,
            normalize_rms=None,
            line_mode=self.line_mode,
        )
        # The model spectrum of THIS render, which is what the old lab drew
        # under its spectrogram; :func:`model_vs_realised` reads it back.
        self.last_diag = diag
        return np.asarray(audio, dtype=np.float64)

    def flight_gain(self, rps_16k: np.ndarray) -> float:
        p = self.params
        exp = p.amp_rps_exponent if p.amp_rps_exponent_floor is None else p.amp_rps_exponent_floor
        return _floor_gain(
            rps_16k,
            ref=float(p.amp_rps_ref),
            exp=float(exp),
            static_rel=float(p.floor_static_rel),
        )

    def describe(self) -> None:
        p = self.params
        prof = np.atleast_2d(np.asarray(p.profile_db, dtype=np.float64))
        print(f"{self.name}  [{self.generation}]  {self.entry}")
        print(
            f"  line_mode {self.line_mode}   n_harmonics {int(p.n_harmonics)}   "
            f"n_rotors {prof.shape[0]}   sample_rate {int(p.sample_rate)}"
        )
        print(
            f"  levels     : harm_mean {float(p.harm_mean_db):+.2f} dB   "
            f"floor_mean {float(p.floor_mean_db):+.2f} dB   "
            f"floor_static_rel {float(p.floor_static_rel):.4g}"
        )
        print(
            f"  speed law  : amp_rps_ref {float(p.amp_rps_ref):.1f} rev/s   "
            f"amp_exp {float(p.amp_rps_exponent):.2f}   floor_exp "
            + (
                "same"
                if p.amp_rps_exponent_floor is None
                else f"{float(p.amp_rps_exponent_floor):.2f}"
            )
        )
        print(
            f"  wander     : harm {float(p.harm_gp_std_db):.2f} dB / "
            f"{float(p.harm_gp_tau_s):.2f} s   coherence {float(p.harm_coherence):.2f}   "
            f"floor {float(p.floor_gp_std_db):.2f} dB / {float(p.floor_gp_tau_s):.2f} s"
        )
        k = min(8, prof.shape[1])
        print(f"  profile_db, k=1..{k} (dB, per rotor):")
        for r in range(prof.shape[0]):
            print(f"    rotor {r}: " + "  ".join(f"{v:7.2f}" for v in prof[r, :k]))


class LegacyRandom(_LegacySource):
    """A FRESH draw of the hand-written family — a new drone every seed.

    :func:`data_processing.stochastic_rotor_noise.sample_params` with the
    family's default ranges, which is what the old ``stochastic_noise_lab.Lab``
    did on "new parameters" and what a ``kind: stochastic`` policy with a
    ``ranges:`` block renders per window.  Keyword overrides are applied on top
    of the draw (``LegacyRandom(seed=3, harm_mean_db=6.0)``), which is what the
    deleted lab's sliders wrote to.
    """

    def __init__(
        self,
        seed: int = 0,
        *,
        ranges: Any = None,
        n_rotors: int = 4,
        n_harmonics: int = 80,
        line_mode: str = LEGACY_RANDOM_LINE_MODE,
        n_fft: int = 2048,
        **overrides: float,
    ):
        self.ranges = ranges
        self.n_rotors = int(n_rotors)
        self.n_harmonics = int(n_harmonics)
        self.overrides = {k: float(v) for k, v in overrides.items()}
        self.seed = int(seed)
        super().__init__(
            self._draw(int(seed)),
            name=self._name(int(seed)),
            entry=self._entry(int(seed)),
            line_mode=line_mode,
            n_fft=n_fft,
        )

    def _draw(self, seed: int):
        from data_processing import stochastic_rotor_noise as srn

        params = srn.sample_params(
            np.random.default_rng(int(seed)),
            self.ranges,
            n_rotors=self.n_rotors,
            n_harmonics=self.n_harmonics,
            sample_rate=SR,
        )
        return params.with_(**self.overrides) if self.overrides else params

    def _name(self, seed: int) -> str:
        return f"legacy-random s{int(seed)}"

    def _entry(self, seed: int) -> str:
        return f"sample_params(seed={int(seed)}, n_harmonics={self.n_harmonics})" + (
            f" + {self.overrides}" if self.overrides else ""
        )

    def resample(self, seed: int | None = None):
        """Draw a NEW drone in place — a new timbre, floor colour and linewidth
        set — and return its parameters.

        The old lab's "New random parameters" button, with its semantics: no
        argument advances the seed by one, so clicking it walks the family.  The
        source keeps its identity (``render_all`` keys by ``name`` at call time,
        and the name carries the new seed).
        """
        self.seed = self.seed + 1 if seed is None else int(seed)
        self.params = self._draw(self.seed)
        self.name = self._name(self.seed)
        self.entry = self._entry(self.seed)
        return self.params


class LegacyBank(_LegacySource):
    """ONE entry of a legacy preset bank — the rigs ``rig_easy`` / ``rig_hard``
    trained on.

    The entry is read through
    :func:`data_processing.stochastic_rotor_noise.load_preset_bank`, the same
    loader :class:`~data_processing.stochastic_rotor_noise.StochasticNoisePool`
    uses for ``preset_bank:``, so the parameter set heard here is bit-for-bit
    the one a training window would have drawn.  The pool's per-window
    ``amp_rps_ref``/``gamma`` rescale by the flight's hover is NOT applied: it
    is part of the policy's speed handling (``rps_scale_range`` and the flight
    cache), not of the rig, and the whole point here is one trajectory with no
    policy in the way.

    :func:`tune` shows an entry's numbers READ-ONLY for the same reason: a bank
    entry is a fitted rig, and a moved number is a different rig than the one
    the arms trained on.  Clip length is the only knob left.
    """

    def __init__(
        self,
        preset: str = "easy",
        index: int = 0,
        *,
        line_mode: str = LEGACY_BANK_LINE_MODE,
        n_fft: int = 2048,
    ):
        key = str(preset).lower()
        if key not in LEGACY_BANKS:
            raise ValueError(
                f"unknown legacy bank {preset!r}; known banks are {list(LEGACY_BANKS)}"
            )
        params = _legacy_entry(key, int(index))
        self.preset = key
        self.index = int(index)
        super().__init__(
            params,
            name=f"legacy-bank {key}[{int(index)}]",
            entry=f"{LEGACY_BANKS[key]}: entry {int(index)}",
            line_mode=line_mode,
            n_fft=n_fft,
        )


def _legacy_entry(preset: str, index: int) -> Any:
    from data_processing import stochastic_rotor_noise as srn

    bank = srn.load_preset_bank(str(ROOT / LEGACY_BANKS[preset]))
    if not 0 <= index < len(bank):
        raise IndexError(f"{LEGACY_BANKS[preset]} holds {len(bank)} entries; asked for {index}")
    return bank[index]


@lru_cache(maxsize=8)
def _v2_entry(preset: str, index: int):
    """ONE entry of a v2 rig bank, through the pool's own bank loader.

    Cached per entry rather than per bank on purpose: a bank is a 30 MB JSON of
    2048 inline fit payloads, and the notebook wants two of them.
    """
    from data_processing.noise_v2_pool import load_preset_bank

    entries = load_preset_bank(str(ROOT / V2_BANKS[preset]))
    if not 0 <= index < len(entries):
        raise IndexError(f"{V2_BANKS[preset]} holds {len(entries)} entries; asked for {index}")
    return entries[index]


class V2Fit(NoiseSource):
    """One of the two WINNING noise-model-v2 fits.

    ``dregon`` is the round-5 UNCALIBRATED single-regime fit rendered by
    :func:`~data_processing.noise_model.render.render_noise` (the +3.75 dB comb
    pin of the calibrated variant is NOT applied; ``comb_offset_db`` puts a
    level offset back by hand).  ``michaels`` is the round-3 standby + cruise
    pair rendered by
    :func:`~data_processing.noise_model.render.render_noise_regimes`, whose
    ``rps_gating`` smoothstep blends the two regimes' POWER between 45 and
    65 rev/s.
    """

    generation = "v2"

    def __init__(self, rig: str = "dregon"):
        self.rig = load_rig(rig)
        self.name = f"v2-fit {self.rig.name}"
        self.entry = " + ".join(self.rig.paths.values())
        self.span = self.rig.span

    def render(
        self,
        rps_16k: np.ndarray,
        *,
        seed: int,
        n_mics: int = 1,
        comb_offset_db: float = 0.0,
    ) -> np.ndarray:
        audio, diag = self.rig.render_audio(
            np.asarray(rps_16k, dtype=np.float64),
            seed=int(seed),
            n_mics=int(n_mics),
            comb_offset_db=float(comb_offset_db),
        )
        self.last_diag = diag
        return audio

    def flight_gain(self, rps_16k: np.ndarray) -> float:
        return _v2_flight_gain(self._level_fit(), rps_16k)

    def _level_fit(self) -> dict[str, Any]:
        """The fit whose floor envelope defines the ``"flight"`` level rule.

        The cruise fit for a per-regime rig — which is the choice
        :meth:`data_processing.noise_v2_pool.NoiseV2Pool._apply_level` makes.
        """
        return self.rig.fits["cruise"] if self.rig.per_regime else self.rig.fits["single"]

    def expected_m(self, rps: np.ndarray, *, n_fft: int, hop: int, comb_offset_db: float = 0.0):
        return self.rig.expected_m(rps, n_fft=n_fft, hop=hop, comb_offset_db=comb_offset_db)

    def describe(self) -> None:
        describe(self.rig)


class V2Bank(NoiseSource):
    """ONE entry of a v2 rig bank (``noise-v2-bank/1``, ``dload:noise-v2-banks``).

    The entry is read by :func:`data_processing.noise_v2_pool.load_preset_bank`
    and rendered exactly as :meth:`~data_processing.noise_v2_pool.NoiseV2Pool.
    render` renders it — ``render_noise_regimes({"standby": ..., "cruise": ...})``
    for a per-regime entry, ``render_noise(entry.cruise)`` otherwise — with the
    pool's entry draw, trajectory draw, level draw and render reuse taken out,
    because those are the policy and not the rig.
    """

    generation = "v2"

    def __init__(self, preset: str = "easy", index: int = 0):
        key = str(preset).lower()
        if key not in V2_BANKS:
            raise ValueError(f"unknown v2 bank {preset!r}; known banks are {list(V2_BANKS)}")
        self.preset = key
        self.index = int(index)
        self.entry_obj = _v2_entry(key, int(index))
        self.name = f"v2-bank {key}[{int(index)}]"
        self.entry = f"{V2_BANKS[key]}: entry {int(index)} ({self.entry_obj.name})"
        self.span = _pool_span(self.entry_obj.cruise)

    @property
    def per_regime(self) -> bool:
        return bool(self.entry_obj.per_regime)

    def fits(self, comb_offset_db: float = 0.0) -> dict[str, dict[str, Any]]:
        """The entry's payloads, keyed as the regime renderer wants them."""
        cruise = _offset_fit(self.entry_obj.cruise, comb_offset_db)
        if not self.per_regime:
            return {"single": cruise}
        return {"standby": _offset_fit(self.entry_obj.standby, comb_offset_db), "cruise": cruise}

    def render(
        self,
        rps_16k: np.ndarray,
        *,
        seed: int,
        n_mics: int = 1,
        comb_offset_db: float = 0.0,
    ) -> np.ndarray:
        from data_processing.noise_model.render import render_noise, render_noise_regimes

        rps = np.asarray(rps_16k, dtype=np.float64)
        fits = self.fits(comb_offset_db)
        if self.per_regime:
            audio = render_noise_regimes(
                {"standby": fits["standby"], "cruise": fits["cruise"]},
                rps,
                sr=SR,
                n_mics=int(n_mics),
                seed=int(seed),
            )
        else:
            audio = render_noise(fits["single"], rps, sr=SR, n_mics=int(n_mics), seed=int(seed))
        return np.asarray(audio, dtype=np.float64)

    def flight_gain(self, rps_16k: np.ndarray) -> float:
        return _v2_flight_gain(self.entry_obj.cruise, rps_16k)

    def expected_m(self, rps: np.ndarray, *, n_fft: int, hop: int, comb_offset_db: float = 0.0):
        from experiments.noise_model import render as RD

        fits = self.fits(comb_offset_db)
        if self.per_regime:
            return RD.expected_periodogram_regimes(
                fits, rps, n_fft=n_fft, hop=hop, sr=SR, n_mics=1, frame_chunk=4
            )
        return RD.expected_periodogram(
            fits["single"], rps, n_fft=n_fft, hop=hop, sr=SR, n_mics=1, frame_chunk=4
        )

    def describe(self) -> None:
        from experiments.noise_model import model as MD

        e = self.entry_obj
        print(f"{self.name}  [{self.generation}]  {self.entry}")
        print(
            f"  regimes    : {'standby + cruise' if self.per_regime else 'cruise only'}   "
            f"traj_rig {e.traj_rig!r}"
        )
        if e.provenance:
            print(f"  provenance : {e.provenance}")
        for regime, fit in self.fits().items():
            p = fit["params"]
            prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
            gamma = np.asarray(MD.gamma_from_params(p), dtype=np.float64)
            fl = p["floor"]
            print(f"  [{regime}] schema {fit.get('schema')}   k_max {prof.shape[1]}")
            print(
                f"    sigma_nu {float(p['sigma_nu']):.4f}   lam {float(p['lam']):.4f}   "
                f"amp_exp {float(p['profile']['amp_exp']):.3f}"
            )
            print(
                f"    floor    : mean {float(fl['floor_mean_db']):.2f} dB   "
                f"tilt {float(fl['floor_tilt_db_oct']):+.3f} dB/oct   "
                f"exp {float(fl['floor_exp']):.3f}   static_rel {float(fl['floor_static_rel']):.4g}"
            )
            k = min(8, prof.shape[1])
            print(f"    profile_db, k=1..{k} (dB, per rotor):")
            for r in range(prof.shape[0]):
                print(f"      rotor {r}: " + "  ".join(f"{v:7.2f}" for v in prof[r, :k]))
            print("    gamma_hz k=1   : " + "  ".join(f"{v:.4f}" for v in gamma[:, 0]))


def _v2_flight_gain(fit: dict[str, Any], rps: np.ndarray) -> float:
    from data_processing.noise_model.constants import AMP_RPS_REF
    from data_processing.noise_model.params import check_schema

    floor = check_schema(fit)["floor"]
    return _floor_gain(
        rps,
        ref=float(AMP_RPS_REF),
        exp=float(floor["floor_exp"]),
        static_rel=float(floor["floor_static_rel"]),
    )


# ── the render ──────────────────────────────────────────────────────────────


def _apply_level(
    audio: np.ndarray,
    source: NoiseSource,
    rps: np.ndarray,
    level: tuple[str, float] | None,
) -> tuple[np.ndarray, float]:
    """``(audio, gain)`` under one of the two level rules, or unchanged."""
    if level is None:
        return audio, 1.0
    mode, value = str(level[0]), float(level[1])
    if mode not in ("window", "flight"):
        raise ValueError(f"level mode must be 'window' or 'flight', got {mode!r}")
    rms = float(np.sqrt(np.mean(np.square(audio)))) or 1.0
    gain = value / rms
    if mode == "flight":
        gain *= source.flight_gain(rps)
    return audio * gain, gain


def render(
    source: NoiseSource,
    traj: td.Frame,
    *,
    seed: int,
    n_mics: int = 1,
    level: tuple[str, float] | None = None,
    comb_offset_db: float = 0.0,
) -> td.Frame:
    """Render ONE source on a trajectory, under one level rule.

    Returns a Frame carrying ``audio`` ``(n_mics, T)`` at 16 kHz, the
    trajectory's own two rotor tracks (``rps`` at 100 Hz for the plots,
    ``rps_render`` at 16 kHz), and a ``meta`` Frame with the trajectory's meta
    plus the source's name, generation and entry, the level rule and the gain it
    applied, and the realised RMS — so ``show``/``line_stats`` and any later
    reader can say exactly what was heard.
    """
    rps = np.asarray(traj["rps_render"].data, dtype=np.float64)
    audio = source.render(rps, seed=int(seed), n_mics=int(n_mics), comb_offset_db=comb_offset_db)
    report = span_report(source.span, rps)
    if source.span is not None and report["frac_below"] + report["frac_above"] > 0.01:
        warnings.warn(
            f"{source.name}: {report['frac_below']:.1%} below / {report['frac_above']:.1%} above "
            f"the fit's carrier span {report['pool_min']:.1f}-{report['pool_max']:.1f} rev/s "
            f"(this trajectory flies {report['rps_min']:.1f}-{report['rps_max']:.1f}). The fitted "
            "speed envelopes extrapolate there. Nothing is clamped.",
            RuntimeWarning,
            stacklevel=2,
        )
    audio, gain = _apply_level(audio, source, rps, level)
    meta = {
        **{k: v for k, v in traj["meta"].items()},
        "source": source.name,
        "generation": source.generation,
        "entry": source.entry,
        "render_seed": int(seed),
        "n_mics": int(n_mics),
        "comb_offset_db": float(comb_offset_db),
        "level_mode": "native" if level is None else str(level[0]),
        "level_value": float("nan") if level is None else float(level[1]),
        "level_gain": float(gain),
        "rms": float(np.sqrt(np.mean(np.square(audio)))),
        "peak": float(np.max(np.abs(audio))),
        **report,
    }
    return td.Frame(
        {
            "audio": td.uniform(audio.astype(np.float32), SR, dims=("mic", "time"), t_start=0.0),
            "rps": traj["rps"],
            "rps_render": traj["rps_render"],
            "meta": td.Frame(meta),
        }
    )


def render_all(
    sources: list[NoiseSource],
    traj: td.Frame,
    *,
    seed: int = 0,
    n_mics: int = 1,
    level: tuple[str, float] | None = None,
    comb_offset_db: float = 0.0,
) -> dict[str, td.Frame]:
    """Every source on the SAME trajectory, seed and level rule.

    Returns ``{source.name: Frame}`` in the order given — what ``show``,
    ``line_stats`` and the players all take.
    """
    out: dict[str, td.Frame] = {}
    for source in sources:
        out[source.name] = render(
            source,
            traj,
            seed=seed,
            n_mics=n_mics,
            level=level,
            comb_offset_db=comb_offset_db,
        )
    return out


# ── looking and listening ───────────────────────────────────────────────────


def player(frame: td.Frame, channel: int = 0):
    """An ``IPython.display.Audio`` widget for one channel of a rendered clip."""
    from IPython.display import Audio

    return Audio(np.asarray(frame["audio"].data)[int(channel)], rate=SR, normalize=True)


def show(
    frames: td.Frame | dict[str, td.Frame],
    *,
    dyn_range: float = 45.0,
    fmax: float = 8000.0,
    figsize: tuple[float, float] = (14, 7),
):
    """``dwym``'s rps route with a READABLE colour range, stacked over clips.

    :func:`plots.dwym.dwym` routes ``audio`` + ``rps`` to a spectrogram over a
    rotor-speed track, which is exactly the figure wanted here, but its
    spectrogram renderer autoscales ``pcolormesh`` over the FULL log-magnitude
    range.  These renders span about 145 dB (a handful of near-empty bins at the
    bottom) while 90 % of the cells live inside 30 dB, so the autoscale washes
    the comb out.  This builds the same tracks through the documented
    track-level escape hatch (``plot_timeframe(frame, tracks=[...])``) with the
    dB data clipped to ``[top - dyn_range, top]``, ``top`` being the 99.5th
    percentile.  Nothing else differs.

    Pass a ``{label: Frame}`` dict — what :func:`render_all` returns — to stack
    several clips in one figure.  The audio widgets are :func:`players`, which
    is a separate call so a figure can be re-drawn without re-listing them.
    """
    from plots.timeframe import PlotTrack, plot_timeframe
    from plots.timeframe.renderers import make_spectrogram_series
    from utils.audio import first_channel

    items = frames if isinstance(frames, dict) else {"": frames}
    tracks: list[Any] = []
    for label, one in items.items():
        spec = make_spectrogram_series(first_channel(one["audio"]), fmax=fmax)
        data = np.asarray(spec.series.data, dtype=np.float32)
        top = float(np.percentile(data, 99.5))
        clipped = td.Series(
            np.clip(data, top - float(dyn_range), top),
            spec.series.dims,
            {"time": spec.series.tindex},
        )
        prefix = f"{label} " if label else ""
        tracks.append(
            PlotTrack(
                series=clipped,
                renderer=spec.renderer,
                hints={**spec.hints, "title": f"{prefix}spectrogram"},
            )
        )
        tracks.append(PlotTrack(series=one["rps"], hints={"title": f"{prefix}rps"}))
    host = next(iter(items.values()))
    fig = plot_timeframe(host, tracks=tracks, figsize=figsize)
    from IPython.display import display

    display(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)


def players(frames: dict[str, td.Frame], channel: int = 0) -> None:
    """One labelled audio widget per clip, without the figure."""
    from IPython.display import display

    for label, one in frames.items():
        m = dict(one["meta"].items())
        print(f"{label}   rms {float(m['rms']):.4g}   peak {float(m['peak']):.4g}")
        display(player(one, channel))


def expected_vs_realised(
    source: NoiseSource,
    frame: td.Frame,
    *,
    n_fft: int = 2048,
    hop: int = 512,
    max_s: float = 4.0,
    f_max: float = 8000.0,
):
    """The v2 fit's expected periodogram against the clip's realised one, 1 mic.

    Both are the campaign's own flight front end (2048 / 512 at 16 kHz) averaged
    over frames, in the absolute units the fit is stated in — so the clip's
    level gain is divided out first, because the level rule is the notebook's
    and not the model's.  Only the first ``max_s`` seconds are used: the forward
    model is the expensive half of the model and a spectrum does not get better
    with more of it.

    Returns a matplotlib Figure.  Legacy sources have no forward model here
    (their expectation is ``stochastic_rotor_noise.model_psd_db`` on the render's
    own diagnostics, a different object), so they are refused by name.
    """
    import matplotlib.pyplot as plt

    expected = getattr(source, "expected_m", None)
    if expected is None:
        raise TypeError(
            f"{source.name} is a {source.generation} source: expected_vs_realised needs a v2 "
            "forward model (V2Fit or V2Bank)"
        )
    rps = np.asarray(frame["rps_render"].data, dtype=np.float64)
    audio = np.asarray(frame["audio"].data, dtype=np.float64)[0]
    n = min(int(round(float(max_s) * SR)), rps.shape[1], audio.size)
    if n < n_fft:
        raise ValueError(f"{n} samples is shorter than n_fft {n_fft}")
    rps, audio = rps[:, :n], audio[:n]
    meta = dict(frame["meta"].items())
    comb_offset_db = float(meta.get("comb_offset_db", 0.0))
    gain = float(meta.get("level_gain", 1.0)) or 1.0
    audio = audio / gain

    m = np.asarray(expected(rps, n_fft=n_fft, hop=hop, comb_offset_db=comb_offset_db))
    model = m[0].mean(axis=0)

    from experiments.stochastic_fit.data import Clip, periodogram

    pg = periodogram(
        Clip("noise_lab", "synthetic", audio[None, :].astype(np.float32), rps, SR),
        n_fft=n_fft,
        hop=hop,
    )
    realised = np.asarray(pg.power, dtype=np.float64)[0].mean(axis=0)

    freqs = np.fft.rfftfreq(n_fft, 1.0 / SR)
    # 30 Hz - f_max: the band the fit's own objective scores. Above ~7.9 kHz the
    # renderer's decimation rolloff drops 80 dB and would own the whole y axis.
    keep = (freqs >= 30.0) & (freqs <= min(float(f_max), 7900.0))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(
        freqs[keep],
        10.0 * np.log10(np.maximum(realised[keep], 1e-300)),
        lw=0.7,
        color="#888888",
        label="realised clip (level gain divided out)",
    )
    ax.plot(
        freqs[keep],
        10.0 * np.log10(np.maximum(model[keep], 1e-300)),
        lw=1.1,
        color="#c0392b",
        label="model expected M",
    )
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power (dB, absolute fitted units)")
    ax.set_title(
        f"{source.name}: expected vs realised periodogram, mic 0, first {n / SR:.1f} s"
        + (f" (comb {comb_offset_db:+.2f} dB)" if comb_offset_db else "")
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    both_db = 10.0 * np.log10(np.maximum(np.concatenate([realised[keep], model[keep]]), 1e-300))
    ax.set_ylim(float(both_db.min()) - 3.0, float(both_db.max()) + 3.0)
    fig.tight_layout()
    return fig


def model_vs_realised(
    source: NoiseSource,
    frame: td.Frame,
    *,
    max_s: float = 2.0,
    f_max: float = 8000.0,
    dyn_range: float = 70.0,
):
    """The model spectrum against the clip's, whichever generation it is.

    A v2 source goes to :func:`expected_vs_realised` — the forward model's
    expected periodogram in absolute fitted units.  A legacy source has no such
    object; what the deleted lab drew instead is the model PSD the render
    ITSELF was built from (``stochastic_rotor_noise.model_psd_db`` on the
    render's diagnostics), frame-averaged, against the realised spectrum, both
    normalised to their own peak — which is why the panel's y axis is relative
    dB for a legacy source and absolute for a v2 one.

    The legacy branch reads the source's MOST RECENT render, so pass the frame
    that render produced.
    """
    if source.generation == "v2":
        return expected_vs_realised(source, frame, max_s=max_s, f_max=f_max)

    import matplotlib.pyplot as plt

    from data_processing import stochastic_rotor_noise as srn

    diag = getattr(source, "last_diag", None)
    if diag is None:
        raise RuntimeError(
            f"{source.name} has not rendered yet — model_vs_realised reads the diagnostics of "
            "the source's last render"
        )
    n_fft = int(getattr(source, "n_fft", 2048))
    audio = np.asarray(frame["audio"].data, dtype=np.float64)
    realised = 10.0 * np.log10(np.maximum(_mean_periodogram(audio, n_fft)[0], 1e-300))
    model = np.asarray(srn.model_psd_db(diag, 0), dtype=np.float64).mean(axis=0)
    if model.size != realised.size:
        raise ValueError(
            f"the stored diagnostics hold {model.size} bins and this clip has {realised.size} — "
            "the frame did not come from this source's last render"
        )
    realised = realised - realised.max()
    model = model - model.max()
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SR)
    keep = freqs <= float(f_max)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(freqs[keep], realised[keep], lw=0.8, color="#888888", label="realised clip")
    ax.plot(freqs[keep], model[keep], lw=1.2, color="#c0392b", label="model PSD of this render")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("dB (each normalised to its own peak)")
    ax.set_title(f"{source.name}: model against realisation, mic 0")
    ax.set_ylim(-float(dyn_range), 3.0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


# ── the number: order-tracked prominence ────────────────────────────────────

#: The R4 "wide" grid the prominence is read on: 8192-point Hann, hop 1024 at
#: 16 kHz (1.95 Hz per bin), exactly ``noise_v2_widen_dregon.WIDE_N/WIDE_HOP``.
LINE_STATS_N_FFT = 8192
LINE_STATS_HOP = 1024


def _mean_periodogram(audio: np.ndarray, n_fft: int) -> np.ndarray:
    """``(M, F)`` frame-averaged periodogram, Hann, 50 % overlap."""
    audio = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    hop = n_fft // 2
    window = np.hanning(n_fft + 1)[:n_fft]
    n_frames = max((audio.shape[-1] - n_fft) // hop + 1, 1)
    if audio.shape[-1] < n_fft:
        audio = np.pad(audio, ((0, 0), (0, n_fft - audio.shape[-1])))
    idx = np.arange(n_frames)[:, None] * hop + np.arange(n_fft)[None, :]
    seg = audio[:, idx] * window  # (M, N, n_fft)
    power = np.abs(np.fft.rfft(seg, axis=-1)) ** 2 / float((window**2).sum())
    return power.mean(axis=1)


def _stft_power(audio: np.ndarray, n_fft: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """``((M, N, F) power, (N,) frame-centre sample index)``.

    ``noise_v2_widen_dregon.stft_power`` verbatim (symmetric Hann, power
    normalised by ``sum(w**2)``, centres at ``start + n_fft // 2``), vectorised
    over frames.
    """
    audio = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    window = np.hanning(n_fft)
    starts = np.arange(0, audio.shape[-1] - n_fft + 1, hop)
    if starts.size == 0:
        raise ValueError(f"{audio.shape[-1]} samples is shorter than one {n_fft}-point frame")
    idx = starts[:, None] + np.arange(n_fft)[None, :]
    seg = audio[:, idx] * window
    power = np.abs(np.fft.rfft(seg, axis=-1)) ** 2 / float((window**2).sum())
    return power, starts + n_fft // 2


def line_stats(
    frames: td.Frame | dict[str, td.Frame],
    *,
    k_max: int = 8,
    n_fft: int = LINE_STATS_N_FFT,
    hop: int = LINE_STATS_HOP,
):
    """Round 4's order-tracked prominence over the local floor, as a table.

    FRAMEWISE, ON EACH FRAME'S OWN CARRIER — which is the whole point of the
    estimator and the one thing a clip-averaged spectrum cannot do.  A fitted
    trajectory moves the rotors over tens of rev/s inside a clip, so a single
    periodogram of the whole clip smears every line across the band it swept
    and reads a width and a prominence that belong to no instant
    (``results/noise_v2/rounds/round2/render_dregon/findings.md``, § "Order-
    tracked comb prominence").  Here each 8192-point Hann frame (hop 1024 at
    16 kHz, 1.95 Hz per bin — ``noise_v2_widen_dregon``'s own wide grid) is
    read at the per-rotor carrier taken from ``rps_render`` AT THAT FRAME'S
    CENTRE SAMPLE.

    The per-frame estimator is R4's, through
    :mod:`experiments.noise_model.tonality`: peak within +-1 bin of ``k f_r``
    over the MEDIAN of the two-sided ``0.45-0.7 fbar`` annulus, with every
    rotor's ``k-1 / k / k+1`` lines excluded from the floor
    (``results/noise_v2/rounds/round4/legacy_truth/anatomy.md``).

    AGGREGATION ORDER, in this order and no other: prominence is computed per
    ``(frame, mic, rotor, k)``, then MEDIAN over frames, then MEDIAN over mics,
    then MEAN over rotors.  The two medians are what keep a single loud frame
    or a single dead microphone out of the number; the rotor mean is R4's
    (``tonality.payload_stats``' ``curve``).

    A frame whose carriers are too slow for the annulus to hold a bin (a
    full-flight trajectory visits the ground) is dropped, and ``n_frames``
    reports how many frames the row was actually read on.

    The difference from :mod:`~experiments.noise_model.tonality` is the input:
    this reads a RENDER, so the numbers carry the renderer's own line shapes
    and its noise, where that module reads the noise-free expected periodogram
    of a payload.

    Returns a ``pandas.DataFrame``: one row per clip, ``k=1 .. k=k_max`` in dB,
    plus the generation, the clip's mean carrier, its RMS and ``n_frames``.
    """
    import pandas as pd

    from experiments.noise_model import tonality as TN

    items = frames if isinstance(frames, dict) else {"": frames}
    df_hz = SR / float(n_fft)
    rows: list[dict[str, Any]] = []
    for label, one in items.items():
        meta = dict(one["meta"].items())
        carrier = np.atleast_2d(np.asarray(one["rps_render"].data, dtype=np.float64))
        audio = np.asarray(one["audio"].data, dtype=np.float64)
        power, centres = _stft_power(audio, int(n_fft), int(hop))
        centres = np.clip(centres, 0, carrier.shape[1] - 1)
        per_frame: list[np.ndarray] = []
        for f, centre in enumerate(centres):
            rev = carrier[:, int(centre)]
            try:
                geom = TN._geometry(rev, power.shape[-1], df_hz, k_geom=int(k_max))
            except ValueError:
                # The annulus holds no bin at this speed: the rotors are too
                # slow for a floor to be read between their orders.
                continue
            per_frame.append(TN.prominence_db(power[:, f, :], geom, int(k_max)))
        if not per_frame:
            raise ValueError(
                f"{label or meta.get('source')}: no frame of this clip turns fast enough for the "
                "0.45-0.7 fbar annulus to hold a bin"
            )
        stack = np.stack(per_frame)  # (N, M, R, K)
        curve = np.nanmean(np.nanmedian(np.nanmedian(stack, axis=0), axis=0), axis=0)  # (K,)
        row: dict[str, Any] = {
            "source": str(meta.get("source", label)),
            "generation": str(meta.get("generation", "?")),
            "fbar_rev_s": float(carrier.mean()),
            "rms": float(meta.get("rms", float("nan"))),
            "n_frames": int(stack.shape[0]),
        }
        row.update({f"k={k + 1}": float(curve[k]) for k in range(int(k_max))})
        rows.append(row)
    return pd.DataFrame(rows).set_index("source")


# ── the panel ───────────────────────────────────────────────────────────────


def _slice_traj(traj: td.Frame, duration_s: float) -> td.Frame:
    """The first ``duration_s`` seconds of a trajectory, both tracks.

    The panel's clip-length knob SLICES the shared trajectory instead of
    drawing a new one — the whole point of this notebook is that every clip
    rides the same carrier, so a length change must not move the aircraft.
    """
    rps = np.asarray(traj["rps_render"].data, dtype=np.float64)
    total = rps.shape[1] / SR
    seconds = float(min(max(float(duration_s), 1.0 / SR), total))
    meta = {**{k: v for k, v in traj["meta"].items()}, "duration_s": seconds}
    return _traj_frame(rps[:, : int(round(seconds * SR))], meta)


def tune(
    source: NoiseSource,
    traj: td.Frame,
    *,
    n_mics: int = 1,
    level: tuple[str, float] | None = ("window", 0.1),
    seed: int = 0,
    dyn_range: float = 70.0,
    model_max_s: float = 2.0,
):
    """The deleted lab's slider panel, over ONE source and the SHARED trajectory.

    What the old ``stochastic_noise_lab.Lab.panel`` was, minus everything the
    merge made shared.  The trajectory controls (rotor-speed source,
    aggressiveness, speed scale, dataset/recording/start) and the level-mode
    dropdown are gone on purpose: the trajectory is the notebook's, drawn once
    by :func:`trajectory`, and the level rule is this call's ``level=``, so
    every clip the panel renders stays comparable with the clips of
    :func:`render_all`.  Everything that describes the NOISE MODEL is still a
    control:

    * :class:`LegacyRandom` — the full :data:`LEGACY_SLIDERS` table (the
      amplitude means, the harmonic and floor wander in dB and in seconds, the
      harmonic coherence, the floor-colour wander and its time, the recording
      floor), the ``line_mode`` dropdown, the clip length, and both buttons.
      A slider move changes THAT NUMBER on the current draw and leaves the
      static random parts — the timbre, the per-line jitter, the floor shape —
      exactly where they were, which is the old semantics; "New random
      parameters" is what redraws them (:meth:`LegacyRandom.resample`).
    * :class:`LegacyBank` — the same table, DISABLED, plus the clip length.
      A bank entry is a FITTED rig: it is the exact parameter set the
      ``rig_easy`` / ``rig_hard`` arms drew from, and moving one of its numbers
      would make it a different rig than the one those arms trained on, which
      is the one question this notebook exists to answer.  The values are shown
      because reading them is the point.
    * :class:`V2Fit` / :class:`V2Bank` — ``comb_offset_db`` (the comb against
      the floor, on a deep copy of the fit) and the clip length.  The v2
      payloads are fitted too; ``comb_offset_db`` is the one knob the campaign
      itself leaves open.

    Every regenerate renders on the same carrier with the same render seed, so
    two panel clips differ only by what was moved.  Returns the ``VBox``; the
    buttons are ``panel.children[1].children[:2]``, and clicking them is how a
    script drives it.
    """
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    legacy = isinstance(source, _LegacySource)
    tunable = isinstance(source, LegacyRandom)
    total_s = float(np.asarray(traj["rps_render"].data).shape[1]) / SR

    style = {"description_width": "170px"}
    layout = widgets.Layout(width="430px")

    def slider(value, lo, hi, step, label, disabled=False):
        return widgets.FloatSlider(
            value=float(np.clip(value, lo, hi)),
            min=lo,
            max=hi,
            step=step,
            description=label,
            continuous_update=False,
            readout_format=".3f",
            disabled=disabled,
            style=style,
            layout=layout,
        )

    duration = slider(
        min(8.0, total_s), min(1.0, total_s), total_s, 0.5, f"clip length (s, of {total_s:.1f})"
    )
    sliders: dict[str, Any] = {}
    line_mode = None
    comb = None
    if legacy:
        sliders = {
            key: slider(getattr(source.params, key), lo, hi, step, label, disabled=not tunable)
            for key, (lo, hi, step, label) in LEGACY_SLIDERS.items()
        }
        line_mode = widgets.Dropdown(
            options=list(LEGACY_LINE_MODES),
            value=source.line_mode,
            description="line mode",
            disabled=not tunable,
            style=style,
            layout=layout,
        )
    else:
        lo, hi, step = COMB_OFFSET_RANGE
        comb = slider(0.0, lo, hi, step, "comb offset (dB)")

    new_params = widgets.Button(
        description="New random parameters", button_style="info", disabled=not tunable
    )
    regenerate = widgets.Button(description="Regenerate", button_style="success")
    status = widgets.HTML()
    out = widgets.Output()
    state: dict[str, Any] = {
        "base": getattr(source, "params", None),
        "frame": None,
        # Until a control is MOVED the source is rendered exactly as it was
        # handed over: a slider whose parameter sits outside its own range
        # would otherwise be clipped into the model on the panel's first draw,
        # and the source object is the caller's, shared with ``render_all``.
        "touched": False,
    }

    def push_sliders() -> None:
        for key, widget in sliders.items():
            widget.value = float(
                np.clip(float(getattr(state["base"], key)), widget.min, widget.max)
            )

    def on_move(_: Any) -> None:
        state["touched"] = True

    def draw(_: Any = None) -> None:
        status.value = "rendering…"
        try:
            sub = _slice_traj(traj, duration.value)
            if tunable and state["touched"]:
                source.params = state["base"].with_(
                    **{key: float(w.value) for key, w in sliders.items()}
                )
                source.line_mode = str(line_mode.value) if line_mode is not None else "stochastic"
            frame = render(
                source,
                sub,
                seed=int(seed),
                n_mics=int(n_mics),
                level=level,
                comb_offset_db=0.0 if comb is None else float(comb.value),
            )
        except Exception as exc:  # noqa: BLE001 — surfaced in the panel, not raised at the user
            status.value = f"<span style='color:#c0392b'>{exc}</span>"
            return
        state["frame"] = frame
        with out:
            clear_output(wait=True)
            import matplotlib.pyplot as plt

            show({source.name: frame}, dyn_range=45.0, figsize=(12, 5))
            fig = model_vs_realised(source, frame, max_s=model_max_s, dyn_range=dyn_range)
            display(fig)
            plt.close(fig)
            display(player(frame))
        m = dict(frame["meta"].items())
        rule = "native" if level is None else f"{level[0]} {level[1]:g}"
        status.value = (
            f"{source.name} — {float(m['duration_s']):.1f} s, level {rule}, "
            f"rms {float(m['rms']):.4g}"
        )

    def on_new(_: Any) -> None:
        state["base"] = source.resample()
        # A fresh drone is a fresh set of numbers: the sliders follow it, and
        # nothing counts as "moved" until the user moves it — hence the reset
        # AFTER push_sliders, whose writes fire the move observer.
        push_sliders()
        state["touched"] = False
        draw()

    for widget in sliders.values():
        widget.observe(on_move, names="value")
    if line_mode is not None:
        line_mode.observe(on_move, names="value")
    new_params.on_click(on_new)
    regenerate.on_click(draw)

    keys = list(sliders)
    half = (len(keys) + 1) // 2
    columns = [
        widgets.VBox([sliders[k] for k in keys[:half]]),
        widgets.VBox([sliders[k] for k in keys[half:]]),
        widgets.VBox([w for w in (line_mode, comb, duration) if w is not None]),
    ]
    controls = widgets.HBox([c for c in columns if c.children])
    draw()
    return widgets.VBox([controls, widgets.HBox([new_params, regenerate, status]), out])


__all__ = [
    "COMB_OFFSET_RANGE",
    "DATASETS",
    "FIT_PATHS",
    "LEGACY_BANKS",
    "LEGACY_LINE_MODES",
    "LEGACY_SLIDERS",
    "RIGS",
    "RPS_PLOT_SR",
    "SR",
    "TRAJ_FITS",
    "TRAJ_FS",
    "TRAJ_KINDS",
    "V2_BANKS",
    "LegacyBank",
    "LegacyRandom",
    "NoiseSource",
    "Rig",
    "V2Bank",
    "V2Fit",
    "describe",
    "describe_traj",
    "expected_vs_realised",
    "line_stats",
    "load_rig",
    "model_vs_realised",
    "player",
    "players",
    "recordings",
    "render",
    "render_all",
    "show",
    "span_report",
    "traj_rig_names",
    "trajectory",
    "tune",
]
