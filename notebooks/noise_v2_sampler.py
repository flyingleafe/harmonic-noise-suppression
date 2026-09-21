"""Sample rendered rotor-noise clips from the two winning noise-model-v2 fits.

The two rigs of the v2 campaign, each driven by the FITTED rotor-speed
trajectory model of the SAME rig, so a clip is a plausible flight of that
aircraft rather than a hand-written speed curve:

* ``dregon`` --- the round-5 UNCALIBRATED single-regime fit
  (``results/noise_v2/rounds/round5/fits/dregon_room2_floor__flight_profile.json``),
  rendered by :func:`experiments.noise_model.render.render_noise`.  The +3.75 dB
  comb pin of the calibrated variant is NOT applied; ``comb_offset_db`` is there
  if you want to put a level offset back by hand.
* ``michaels`` --- the round-3 PER-REGIME candidate, a standby fit plus a cruise
  fit (``.../round3/fits/michaels_fly125_{standby,cruise}__flight.json``),
  rendered by :func:`experiments.noise_model.render.render_noise_regimes`, whose
  ``rps_gating`` smoothstep blends the two regimes' POWER between 45 and
  65 rev/s.

Trajectories come from :mod:`data_processing.trajectory_model` --- the same
``rps-traj-fits`` bundle the ``fitted_traj`` training streams read
(``conf/online_mix/traj_fitted_5050.yaml``), with ``measurement_noise=False``
because the audio is rendered FROM these labels and must not hear the ESC's
sample-and-hold.

Each rig flies its OWN fitted trajectory by default.  ``traj_rig=`` overrides
that with any rig in the bundle, or with ``"posterior"`` --- the reserved name
that draws a FRESH drone from the global rig hyperprior for the flight (its own
``mu``, trim, dynamics, level-offset model, idle level and ESC clamp).  A
hyperprior drone hovers anywhere from ~40 to ~300 rev/s (5th-95th percentile;
median ~137), so it routinely leaves the noise fit's carrier span in one
direction or the other; :func:`sample_trajectory` WARNS with both ranges and
clamps nothing, and :func:`describe_draw` prints which drone was drawn.

The public entry points, in the order the notebook calls them::

    rig   = load_rig("dregon")                       # fits + provenance
    describe(rig)                                    # the fit's key numbers
    traj  = sample_trajectory(rig, seed=0, duration_s=20.0)
    describe_draw(traj)                              # which drone, and its span
    frame = render(rig, traj, seed=0, n_mics=1)      # audio + rps in one Frame
    dwym(frame)                                      # spectrogram + rps overlay
    expected_vs_realised(rig, frame)                 # model M vs the clip

THE RATE CONTRACT.  :func:`render_noise` wants ``rps_rev_s`` as ``(R, T)`` on
its OWN OUTPUT GRID --- one carrier sample per audio sample at ``sr`` (16 kHz),
not at a label rate --- and interpolates that to ``sr_work`` (64 kHz) itself.
The trajectory fits are stated on a 100 Hz grid (``params.RATE_HZ``) and their
whole content lives below 20 Hz, so :func:`sample_trajectory` draws on that grid
and linearly interpolates to 16 kHz, which is exactly what the renderer would
have done internally anyway.  The rendered Frame therefore carries TWO rotor
tracks: ``rps`` decimated to 100 Hz (what the plots draw) and ``rps_render``,
the full 16 kHz carrier the renderer actually consumed.

MEMORY.  Everything here is rendering only; no forward model is ever evaluated
on a fit POOL.  A 30 s 8-mic clip is the largest thing the notebook builds.
"""

from __future__ import annotations

import copy
import json
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td

ROOT = Path(__file__).resolve().parent.parent

#: The renderer's output rate, and therefore the rate its ``rps_rev_s`` is on.
SR = 16000

#: The grid the trajectory fits are stated on (``params.RATE_HZ``).
TRAJ_FS = 100.0

#: Rate the plotted ``rps`` track is carried at.  The trajectory model's fastest
#: component is a 20 Hz oscillator, so 100 Hz loses nothing visible.
RPS_PLOT_SR = 100

#: The trajectory fits bundle (``dload pull rps-traj-fits``).
TRAJ_FITS = "dload:rps-traj-fits"

#: The noise fits of each rig, by regime.  ``"single"`` is a one-regime rig.
FIT_PATHS: dict[str, dict[str, str]] = {
    "dregon": {
        "single": "results/noise_v2/rounds/round5/fits/dregon_room2_floor__flight_profile.json",
    },
    "michaels": {
        "standby": "results/noise_v2/rounds/round3/fits/michaels_fly125_standby__flight.json",
        "cruise": "results/noise_v2/rounds/round3/fits/michaels_fly125_cruise__flight.json",
    },
}

#: Trajectory-model rig name per noise rig.
TRAJ_RIG = {"dregon": "dregon", "michaels": "michaels"}

RIGS = tuple(FIT_PATHS)


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


# ── the rig ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Rig:
    """One rig: its fit(s), how to render them, and where they came from."""

    name: str
    #: ``{"single": fit}`` or ``{"standby": fit, "cruise": fit}``.
    fits: dict[str, dict[str, Any]]
    #: Rig name inside the ``rps-traj-fits`` bundle.
    traj_rig: str
    #: ``{regime: repo-relative fit path}``.
    paths: dict[str, str]
    #: ``{regime: the fit's own recorded carrier span (rev/s)}``.
    pool_rps: dict[str, tuple[float, float] | None]
    #: ``{regime: (schema, git sha the fit was written at)}``.
    provenance: dict[str, dict[str, str]]
    #: Short SHA of this checkout, when the rig was loaded.
    repo_sha: str

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
        out: dict[str, dict[str, Any]] = {}
        for regime, fit in self.fits.items():
            copied = copy.deepcopy(fit)
            prof = np.asarray(copied["params"]["profile"]["profile_db"], dtype=np.float64)
            copied["params"]["profile"]["profile_db"] = (prof + float(comb_offset_db)).tolist()
            out[regime] = copied
        return out

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
    """Load one rig's noise fit(s) with their provenance.

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


# ── the trajectory ──────────────────────────────────────────────────────────


def traj_rig_names() -> tuple[str, ...]:
    """Every trajectory the bundle offers: the seven fitted rigs plus
    ``"posterior"``, the reserved name that draws a FRESH drone from the global
    rig hyperprior (``posterior.json``) for every flight."""
    from data_processing.trajectory_model.source import load_bundle

    return tuple(load_bundle(TRAJ_FITS).names)


def _traj_source(rig: Rig, *, traj_rig: str, mean_shift: float, mean_scale: float):
    """A one-rig :class:`FittedTrajectorySource` with DETERMINISTIC level knobs.

    ``mean_shift`` / ``mean_scale`` are ranges in the training policy; here they
    are single numbers, passed as a degenerate range so a seed pins the clip.
    ``traj_rig`` is any name in the bundle, including ``"posterior"``.
    """
    from data_processing.trajectory_model.source import FittedTrajectorySource, load_bundle

    bundle = load_bundle(TRAJ_FITS)
    if traj_rig not in bundle.names:
        raise ValueError(f"unknown trajectory rig {traj_rig!r}; bundle offers {bundle.names}")
    return FittedTrajectorySource(
        bundle,
        {traj_rig: 1.0},
        mean_shift=(float(mean_shift), float(mean_shift)),
        mean_scale=(float(mean_scale), float(mean_scale)),
        measurement_noise=False,
    )


def span_report(rig: Rig, rps: np.ndarray) -> dict[str, float]:
    """How far a carrier track leaves the span the fit was identified on."""
    lo, hi = rig.span
    rps = np.asarray(rps, dtype=np.float64)
    return {
        "rps_min": float(rps.min()),
        "rps_max": float(rps.max()),
        "pool_min": float(lo),
        "pool_max": float(hi),
        "frac_below": float(np.mean(rps < lo)),
        "frac_above": float(np.mean(rps > hi)),
    }


def sample_trajectory(
    rig: Rig,
    *,
    seed: int,
    duration_s: float,
    mean_shift: float = 0.0,
    mean_scale: float = 1.0,
    full_flight: bool = False,
    traj_rig: str | None = None,
    fs: float = TRAJ_FS,
) -> td.Frame:
    """One trajectory of a fitted rotor-speed model, as a Frame.

    Parameters
    ----------
    seed
        Seeds the trajectory draw only; the render has its own seed.
    duration_s
        Clip length in seconds.  With ``full_flight`` this is the WHOLE flight,
        ground to ground, so the airborne share is shorter.
    mean_shift, mean_scale
        Move the rig's hover level: ``mu * scale + shift`` (rev/s).  The ESC
        clamp and the warm-up idle level travel with it, exactly as in
        ``conf/online_mix/traj_fitted_5050.yaml``.
    full_flight
        ``True`` wraps the airborne process in
        :func:`data_processing.trajectory_model.flight.wrap_airborne` --- ground
        silence, spin-up, warm-up idle, take-off, airborne, landing, spin-down.
        For ``michaels`` that is what drives the carrier through the 45-65 rev/s
        band and exercises the standby/cruise blend.
    traj_rig
        Which trajectory to fly.  Defaults to the noise rig's OWN fitted rig
        (``rig.traj_rig``).  Any name in :func:`traj_rig_names` works --- the
        seven fitted rigs, or ``"posterior"``, the reserved name that draws a
        FRESH drone from the global rig hyperprior (each draw gets its own
        per-rig level-offset model, its own idle level and its own ESC clamp,
        exactly as the training stream's ``rigs: {posterior: 1}`` does).  A
        posterior drone hovers anywhere from ~40 to ~300 rev/s (5th-95th
        percentile; median ~137), so it usually leaves the noise fit's carrier
        span in one direction or the other; the span warning below says so, in
        both ranges, and NOTHING is clamped.
    fs
        Rate the state space is simulated on before interpolation to 16 kHz.
        The shipped fits are stated on 100 Hz; there is no reason to change it.

    Returns
    -------
    A Frame with ``rps`` --- a ``(4, duration_s * 16000)`` Series in rev/s ON
    THE RENDERER'S GRID (see the module docstring) --- plus a ``meta`` Frame
    carrying the DRAWN DRONE (which rig, its per-rotor ``mu``, hover level,
    warm-up idle, ESC clamp and per-flight offset stds) and the span report.
    """
    name = rig.traj_rig if traj_rig is None else str(traj_rig)
    src = _traj_source(rig, traj_rig=name, mean_shift=mean_shift, mean_scale=mean_scale)
    rng = np.random.default_rng(int(seed))
    n_low = int(round(float(duration_s) * float(fs)))
    if full_flight:
        low = src.flight(rng, float(fs), duration_s=float(duration_s))
    else:
        draw = src.draw(rng)
        src.last_draw = draw
        low = draw.params.sample_airborne(n_low, rng, fs=float(fs), clip=draw.clip)
    low = np.atleast_2d(np.asarray(low, dtype=np.float64))

    n_out = int(round(float(duration_s) * SR))
    t_low = np.arange(low.shape[1]) / float(fs)
    t_out = np.arange(n_out) / float(SR)
    rps = np.stack([np.interp(t_out, t_low, low[r]) for r in range(low.shape[0])])

    report = span_report(rig, rps)
    drawn = src.last_draw
    who = f"{name} drone" if name != "posterior" else "posterior-drawn drone"
    span_line = (
        f"the {rig.name} fit's pool span is {report['pool_min']:.1f}-{report['pool_max']:.1f} "
        f"rev/s; this {who} flies {report['rps_min']:.1f}-{report['rps_max']:.1f} rev/s"
    )
    if report["frac_above"] > 0.0:
        warnings.warn(
            f"{report['frac_above']:.1%} of the sampled carrier is ABOVE the fit's span — "
            f"{span_line}. The fitted speed envelopes (profile.amp_exp, floor.floor_exp) "
            "extrapolate there. Nothing is clamped.",
            RuntimeWarning,
            stacklevel=2,
        )
    if report["frac_below"] > 0.0 and (not full_flight or name != rig.traj_rig):
        warnings.warn(
            f"{report['frac_below']:.1%} of the sampled carrier is BELOW the fit's span — "
            f"{span_line}. The speed envelopes extrapolate there too"
            + (" (a full flight visits the ground on purpose)." if full_flight else ".")
            + " Nothing is clamped.",
            RuntimeWarning,
            stacklevel=2,
        )
    meta = {
        "rig": rig.name,
        "traj_rig": name,
        "traj_rig_drawn": str(drawn.rig) if drawn is not None else None,
        "seed": int(seed),
        "duration_s": float(duration_s),
        "mean_shift": float(mean_shift),
        "mean_scale": float(mean_scale),
        "full_flight": bool(full_flight),
        "traj_fs": float(fs),
        "measurement_noise": False,
        "esc_clip_rev_s": list(drawn.clip) if drawn is not None else None,
        "mu_rev_s": drawn.params.mu.tolist() if drawn is not None else None,
        "hover_rev_s": float(np.mean(drawn.params.mu)) if drawn is not None else None,
        "idle_rev_s": np.asarray(drawn.idle_rps).tolist() if drawn is not None else None,
        "offset_s_c": float(drawn.params.s_c) if drawn is not None else None,
        "offset_s_r": np.asarray(drawn.params.s_r).tolist() if drawn is not None else None,
        **{k: float(v) for k, v in report.items()},
    }
    return td.Frame(
        {
            "rps": td.uniform(rps, SR, dims=("rotor", "time"), t_start=0.0),
            "meta": td.Frame(meta),
        }
    )


# ── the render ──────────────────────────────────────────────────────────────


def render(
    rig: Rig,
    traj: td.Frame,
    *,
    seed: int,
    n_mics: int = 1,
    comb_offset_db: float = 0.0,
) -> td.Frame:
    """Render ``rig``'s fitted noise on ``traj``'s carrier.

    ``comb_offset_db`` is added to every rotor's ``profile_db`` of a DEEP COPY
    of the fit (the loaded fits are never touched); 0 is the uncalibrated rig
    as fitted.  For ``michaels`` the standby and cruise fits are composed by
    :func:`experiments.noise_model.render.render_noise_regimes`.

    Returns a Frame carrying ``audio`` ``(n_mics, T)`` at 16 kHz in ABSOLUTE
    fitted units (no RMS normalisation), ``rps`` decimated to 100 Hz for the
    plots, ``rps_render`` --- the exact 16 kHz carrier the renderer consumed ---
    and a ``meta`` Frame with the render diagnostics, so ``plots.dwym(frame)``
    draws the spectrogram with the rotor speeds under it.
    """
    rps = np.asarray(traj["rps"].data, dtype=np.float64)
    audio, diag = rig.render_audio(rps, seed=seed, n_mics=n_mics, comb_offset_db=comb_offset_db)
    step = max(int(round(SR / RPS_PLOT_SR)), 1)
    meta = {
        **{k: v for k, v in traj["meta"].items()},
        "render_seed": int(seed),
        "n_mics": int(n_mics),
        "comb_offset_db": float(comb_offset_db),
        "rms": [float(v) for v in diag["rms"]],
        "per_regime": rig.per_regime,
        "repo_sha": rig.repo_sha,
    }
    if rig.per_regime:
        w = diag["cruise_weight"]
        meta["cruise_weight"] = [float(w["min"]), float(w["max"]), float(w["mean"])]
        meta["frac_in_blend"] = float(w["fraction_in_blend"])
    return td.Frame(
        {
            "audio": td.uniform(audio.astype(np.float32), SR, dims=("mic", "time"), t_start=0.0),
            "rps": td.uniform(
                np.ascontiguousarray(rps[:, ::step]),
                RPS_PLOT_SR,
                dims=("rotor", "time"),
                t_start=0.0,
            ),
            "rps_render": traj["rps"],
            "meta": td.Frame(meta),
        }
    )


def sample_clip(
    rig_name: str,
    *,
    seed: int = 0,
    duration_s: float = 20.0,
    mean_shift: float = 0.0,
    mean_scale: float = 1.0,
    full_flight: bool = False,
    traj_rig: str | None = None,
    n_mics: int = 1,
    comb_offset_db: float = 0.0,
    render_seed: int | None = None,
) -> td.Frame:
    """The whole knob panel in one call: load, sample, render.

    ``render_seed`` defaults to ``seed``, so one number moves both draws.
    ``traj_rig`` defaults to the noise rig's own trajectory fit; pass
    ``"posterior"`` to fly a drone drawn from the rig hyperprior.
    """
    rig = load_rig(rig_name)
    traj = sample_trajectory(
        rig,
        seed=seed,
        duration_s=duration_s,
        mean_shift=mean_shift,
        mean_scale=mean_scale,
        full_flight=full_flight,
        traj_rig=traj_rig,
    )
    return render(
        rig,
        traj,
        seed=seed if render_seed is None else int(render_seed),
        n_mics=n_mics,
        comb_offset_db=comb_offset_db,
    )


def describe_draw(frame: td.Frame) -> None:
    """Print WHICH drone a trajectory Frame (or a rendered Frame) was flown on,
    and how its speeds sit against the noise fit's own carrier span."""
    m = dict(frame["meta"].items())
    drawn = m.get("traj_rig_drawn")
    asked = m.get("traj_rig")
    origin = "the rig's own fit" if drawn != "posterior" else "a FRESH draw from the hyperprior"
    print(f"trajectory   : {asked}  ({origin})")
    mu = [float(v) for v in (m.get("mu_rev_s") or [])]
    print(
        f"  hover      : {float(m['hover_rev_s']):.1f} rev/s   "
        f"mu " + " ".join(f"{v:.1f}" for v in mu)
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
    print(
        f"  flown      : {float(m['rps_min']):.1f}-{float(m['rps_max']):.1f} rev/s   "
        f"vs the {m['rig']} fit's pool span {float(m['pool_min']):.1f}-"
        f"{float(m['pool_max']):.1f} rev/s"
    )
    below, above = float(m["frac_below"]), float(m["frac_above"])
    if below or above:
        print(
            f"  EXTRAPOLATING: {below:.1%} below / {above:.1%} above the fitted span — "
            "the speed envelopes (amp_exp, floor_exp) are outside their evidence. "
            "Nothing is clamped."
        )
    else:
        print("  the whole trajectory is inside the span the fit was identified on.")


def player(frame: td.Frame, channel: int = 0):
    """An ``IPython.display.Audio`` widget for one channel of a rendered clip."""
    from IPython.display import Audio

    return Audio(np.asarray(frame["audio"].data)[int(channel)], rate=SR, normalize=True)


def show(
    frame: td.Frame | dict[str, td.Frame],
    *,
    dyn_range: float = 45.0,
    fmax: float = 8000.0,
    figsize: tuple[float, float] = (14, 7),
):
    """``dwym``'s rps route with a READABLE colour range.

    :func:`plots.dwym.dwym` routes ``audio`` + ``rps`` to a spectrogram over a
    rotor-speed track, which is exactly the figure wanted here, but its
    spectrogram renderer autoscales ``pcolormesh`` over the FULL log-magnitude
    range.  These renders span about 145 dB (a handful of near-empty bins at the
    bottom) while 90 % of the cells live inside 30 dB, so the autoscale washes
    the comb out.  This builds the same tracks through the documented
    track-level escape hatch (``plot_timeframe(frame, tracks=[...])``) with the
    dB data clipped to ``[top - dyn_range, top]``, ``top`` being the 99.5th
    percentile.  Nothing else differs.

    Pass a ``{label: Frame}`` dict to stack several clips in one figure.
    """
    from plots.timeframe import PlotTrack, plot_timeframe
    from plots.timeframe.renderers import make_spectrogram_series
    from utils.audio import first_channel

    frames = frame if isinstance(frame, dict) else {"": frame}
    tracks: list[Any] = []
    for label, one in frames.items():
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
    host = next(iter(frames.values()))
    fig = plot_timeframe(host, tracks=tracks, figsize=figsize)
    from IPython.display import display

    display(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)
    for one in frames.values():
        display(player(one))


# ── model against realisation ───────────────────────────────────────────────


def expected_vs_realised(
    rig: Rig,
    frame: td.Frame,
    *,
    n_fft: int = 2048,
    hop: int = 512,
    max_s: float = 4.0,
    f_max: float = 8000.0,
):
    """The fit's expected periodogram against the clip's realised one, 1 mic.

    Both are the campaign's own flight front end (2048 / 512 at 16 kHz) averaged
    over frames, in the absolute units the fit is stated in.  Only the first
    ``max_s`` seconds are used: the forward model is the expensive half of the
    model and a spectrum does not get better with more of it.

    Returns a matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    rps = np.asarray(frame["rps_render"].data, dtype=np.float64)
    audio = np.asarray(frame["audio"].data, dtype=np.float64)[0]
    n = min(int(round(float(max_s) * SR)), rps.shape[1], audio.size)
    if n < n_fft:
        raise ValueError(f"{n} samples is shorter than n_fft {n_fft}")
    rps, audio = rps[:, :n], audio[:n]
    comb_offset_db = float(frame["meta"]["comb_offset_db"])

    m = np.asarray(rig.expected_m(rps, n_fft=n_fft, hop=hop, comb_offset_db=comb_offset_db))
    model = m[0].mean(axis=0)

    from experiments.stochastic_fit.data import Clip, periodogram

    pg = periodogram(
        Clip("noise_v2_sampler", "synthetic", audio[None, :].astype(np.float32), rps, SR),
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
        label="realised clip",
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
        f"{rig.name}: expected vs realised periodogram, mic 0, first {n / SR:.1f} s"
        + (f" (comb {comb_offset_db:+.2f} dB)" if comb_offset_db else "")
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    both_db = 10.0 * np.log10(np.maximum(np.concatenate([realised[keep], model[keep]]), 1e-300))
    ax.set_ylim(float(both_db.min()) - 3.0, float(both_db.max()) + 3.0)
    fig.tight_layout()
    return fig


# ── the readout ─────────────────────────────────────────────────────────────


def describe(rig: Rig, *, k_show: int = 8) -> None:
    """Print the fit's key numbers, per regime, plus its provenance."""
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


__all__ = [
    "FIT_PATHS",
    "RIGS",
    "RPS_PLOT_SR",
    "SR",
    "TRAJ_FITS",
    "TRAJ_FS",
    "Rig",
    "describe",
    "describe_draw",
    "expected_vs_realised",
    "load_rig",
    "player",
    "render",
    "sample_clip",
    "sample_trajectory",
    "show",
    "span_report",
    "traj_rig_names",
]
