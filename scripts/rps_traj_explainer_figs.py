#!/usr/bin/env python3
"""Every figure of ``docs/explainers/rps-trajectory-model.qmd``.

Reads ONLY the campaign's own artifacts — ``results/rps_traj/`` (fits,
statistics, discrepancy summary, posterior + draws, exploration PNGs) and the
published frames datasets through :func:`experiments.rps_traj.data.load_rig` —
and writes PNGs plus ``figure_index.json`` into the explainer's figure folder.
The qmd loops over that index, so a round re-fit is
``python scripts/rps_traj_explainer_figs.py && quarto render …`` with no hand
edits anywhere.

Per rig (``data.RIGS``, or ``--rigs``):

1. ``<rig>_zoom.png`` — 20 s of the four rotors, real / fitted current sampler
   / fitted new model, one shared y-range.
2. ``<rig>_flight.png`` — a WHOLE real recording (ground, spin-up, idle,
   take-off, airborne, landing) against ``full_flight`` of both fits.
3. ``<rig>_stats.png`` — the frozen statistics themselves: per-rotor ACF on
   ``stats.LAGS_S``, the three 4x4 cross-correlation matrices, and the five
   discrepancy families of both arms.
4. ``<rig>_spectra.png`` — per-mode Welch spectra (real, base samples, new
   samples, new analytic) on the fit's own block geometry.

From the posterior: ``posterior_zoom_grid.png`` (``--n-draws`` drawn drones,
20 s each), ``posterior_flight_grid.png`` (six whole flights) and
``posterior_planes.png`` (the seven fitted rigs and the draws in three
parameter planes).

The block recipe of figure 4 is a copy of ``scripts/_rps_traj_diag.py``'s
``mode_periodogram`` rewritten to take segments (so the same code path serves
real segments and model samples); the diagnostic script itself is deliberately
NOT imported — it is a one-off with its own CLI.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np  # noqa: E402
from matplotlib.axes import Axes
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
from scipy.signal.windows import hann  # noqa: E402

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))

from data_processing.trajectory_model import NewFit, Params, Posterior  # noqa: E402
from experiments.rps_traj.baseline import BaselineFit  # noqa: E402
from experiments.rps_traj.data import (  # noqa: E402
    RATE_HZ,
    RIGS,
    Flight,
    airborne_segments,
    ground_level,
    load_rig,
)
from experiments.rps_traj.posterior import fit_posterior  # noqa: E402
from experiments.rps_traj.stats import (  # noqa: E402
    FAMILIES,
    LAGS_S,
    TrajStats,
    discrepancy,
    passes,
)
from tracking.rotors import MIXER, MODE_NAMES, NUM_ROTORS  # noqa: E402

#: Rotors in MIXER order, with the colour every figure uses for them.
ROTOR_NAMES = ("RFront", "LFront", "LBack", "RBack")
ROTOR_SHORT = ("RF", "LF", "LB", "RB")
ROTOR_COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd")

#: The three arms, with their colour and the label the figures use.
ARMS = ("real", "base", "new")
ARM_LABEL = {
    "real": "real telemetry",
    "base": "current sampler, fitted",
    "new": "new model, fitted",
}
ARM_COLOR = {"real": "#111111", "base": "#e8790c", "new": "#1f77b4"}

#: Mode colours of the spectra figure (same as the diagnostic script's).
MODE_COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd")

#: Welch geometry of the mode-spectra figure: 20 s Hann blocks at 50 %
#: overlap, whole-segment blocks down to 5 s. These are the round-3 Whittle
#: block constants, copied here ON PURPOSE rather than imported: the
#: likelihood's own geometry is a modelling choice that moves between rounds
#: (round 4 replaced the Whittle blocks with an exact Kalman recursion), while
#: a PLOT of the spectra has to keep one fixed recipe or the panels of two
#: rounds stop being comparable.
BLOCK_S = 20.0
BLOCK_OVERLAP = 0.5
MIN_BLOCK_S = 5.0

#: Zoomed window (s) and the frequency band of the spectra figure (Hz).
ZOOM_S = 20.0
SPECTRA_BAND = (0.02, 40.0)

#: Cap on the simulated seconds per arm behind one spectra figure. The
#: baseline generator runs at ~2000x realtime, so this is a second of compute;
#: the cap exists so that neurobem's 2479 airborne seconds do not quietly make
#: the figure cost 10x every other rig's.
MODEL_SIM_CAP_S = 2000.0

#: Whole-flight grid: how many posterior draws, and the floor on a drawn
#: flight's length (s) so the fixed phases are not squeezed to nothing.
N_FLIGHT_DRAWS = 6
MIN_FLIGHT_S = 60.0

DPI = 110

#: Exploration PNGs copied into the figure folder, with the finding each one
#: carries (``results/rps_traj/explore/findings.md``).
EXPLORE_FIGS: tuple[tuple[str, str], ...] = (
    (
        "dregon_channel_fidelity.png",
        "DREGON's `motor.measured` is a ~45 Hz sample-and-hold, not a 1002 Hz "
        "measurement: 95.3 % of consecutive native samples are bit-identical and "
        "the measured-minus-command residual accounts for 106 % of the 5-15 Hz "
        "band variance. Above ~5 Hz the channel shows its own staircase.",
    ),
    (
        "michaels_quant.png",
        "Michael's telemetry has the opposite defect: 1 RPM amplitude steps "
        "(0.02 % of the operating point) but only a 29.41 Hz log rate, partly "
        "held (P(delta = 0) ~ 0.42, median hold 68 ms). The 12-14.7 Hz band sits "
        "2000x above the quantisation floor, so the near-Nyquist content is "
        "folding and hold-resampling, not shaft motion.",
    ),
    (
        "michaels_psd.png",
        "A DJI M100's rotor speed is a nearly featureless power law: the best "
        "hump in the band is +2.0 dB at 8.70 Hz with no -3 dB crossing inside "
        "the band, broadband slope -20.9 dB/decade.",
    ),
    (
        "dregon_room2_command_psd.png",
        "The only clean resonance in the corpus is in the DREGON command "
        "channel: +4.3 dB at 3.00 Hz, -3 dB width 2.80-3.30 Hz (Q ~ 6), second "
        "peak 6.7 dB. That is the MikroKopter attitude loop. A resonance must "
        "therefore be a per-rig parameter that is allowed to vanish.",
    ),
    (
        "michaels_eigen.png",
        "The four control modes do not share a spectral shape. The collective "
        "and yaw modes carry 78 % of the rotor variance, roll and pitch sit "
        "~10 dB below across the whole band, and the leading eigenvector is a "
        "collective/yaw mixture (cosines 0.55 / 0.67), not a mixer column.",
    ),
    (
        "michaels_acf.png",
        "The autocorrelation has two time scales — 0.98 at 0.02 s, 0.82 at "
        "0.1 s, 0.16 at 1 s, 0.007 at 10 s, 1/e crossing 0.43 s — so no single "
        "exponential (AR(1)) fits it.",
    ),
)


# ─── small helpers ────────────────────────────────────────────────────────────


def _save(fig: Figure, out: Path, name: str) -> str:
    # `optimize` is lossless and worth 10-20 % on these line-heavy panels; the
    # explainer embeds every figure as base64, which costs another third.
    fig.savefig(
        out / name,
        dpi=DPI,
        bbox_inches="tight",
        pil_kwargs={"optimize": True, "compress_level": 9},
    )
    plt.close(fig)
    return name


def _rig_label(rig: str) -> str:
    return rig.replace("_quad", "").replace("_", " ")


def _segments(flight: Flight) -> list[slice]:
    return airborne_segments(flight.rps, flight.fs)


def _airborne_s(flight: Flight) -> float:
    return sum(sl.stop - sl.start for sl in _segments(flight)) / float(flight.fs)


def _zoom_window(flights: list[Flight], seconds: float) -> tuple[Flight, int, int]:
    """``(flight, start, n)``: ``seconds`` from the middle of the longest
    airborne segment of ``flights`` (the whole segment when it is shorter)."""
    best: tuple[int, Flight, slice] | None = None
    for flight in flights:
        for sl in _segments(flight):
            n = int(sl.stop - sl.start)
            if best is None or n > best[0]:
                best = (n, flight, sl)
    if best is None:  # no airborne segment anywhere: show the middle of the log
        flight = max(flights, key=lambda f: f.rps.shape[1])
        n = min(int(round(seconds * flight.fs)), flight.rps.shape[1])
        start = (flight.rps.shape[1] - n) // 2
        return flight, start, n
    total, flight, sl = best
    want = min(int(round(seconds * flight.fs)), total)
    start = int(sl.start) + (total - want) // 2
    return flight, start, want


def _ground_flight(flights: list[Flight]) -> Flight:
    """The recording that best shows a WHOLE flight: one with a pre-take-off
    idle plateau if the source has any, the longest such, else the longest."""
    with_ground = [f for f in flights if ground_level(f) is not None]
    pool = [f for f in with_ground if f.duration_s >= 30.0] or with_ground or list(flights)
    return max(pool, key=lambda f: f.duration_s)


def _rotor_lines(ax: Axes, t: np.ndarray, rps: np.ndarray, lw: float = 0.8) -> None:
    for i in range(NUM_ROTORS):
        ax.plot(t, rps[i], lw=lw, color=ROTOR_COLORS[i], label=ROTOR_NAMES[i])


def _finite_range(
    arrays: list[np.ndarray], pad: float = 0.04, robust: bool = False
) -> tuple[float, float]:
    """Shared y-range over several series.

    With ``robust`` the range covers the 0.2 to 99.8 percentile instead of the
    full extent. One logger glitch — blackbird ends its recording with a single
    1100 rev/s sample — otherwise flattens a whole row of a grid. Rare samples
    then fall outside the axes, and the caption says so.
    """
    vals = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    if robust and vals.size > 100:
        lo, hi = (float(v) for v in np.percentile(vals, [0.2, 99.8]))
    else:
        lo, hi = float(vals.min()), float(vals.max())
    span = max(hi - lo, 1e-6)
    return lo - pad * span, hi + pad * span


#: Sentence appended to a figure caption whose new-model panel is absent.
MISSING_ARM_NOTE = (
    " The new-model panel is ABSENT here: the stored fit under `fits/new/` was "
    "written by an earlier round and its parameter set is not a re-expressible "
    "one, so no sample can be drawn from it. Re-run the fit and this figure "
    "gains its third panel with no other change."
)

#: ``tau_e`` of a re-expressed round-3 fit (s), and the label that travels with
#: it. Round 3's measurement term was WHITE with std ``sigma_w``; the round-4
#: family contains that exact process as the ``tau_e -> 0`` limit at
#: ``sigma_e = sigma_w``. 1 ms is the parametrisation's own floor and leaves a
#: lag-1 correlation of ``exp(-10) = 4.5e-5`` on the 100 Hz grid, i.e. white to
#: four decimals. This is a RE-EXPRESSION of a stored fit, never a new fit.
LEGACY_TAU_E_S = 1e-3
LEGACY_NOTE = (
    " The new-model arm here comes from an **earlier round** of the fit, not "
    "from the round the text describes: `fits/new/` has not been re-run yet. "
    "The stored parameters are re-expressed exactly in the current "
    "parameterisation, so the panel is a faithful picture of that earlier fit."
)


def _missing_arm_note(new: NewFit | None) -> str:
    return MISSING_ARM_NOTE if new is None else str(getattr(new, "legacy_note", ""))


def _arm_label(arm: str, new: NewFit | None) -> str:
    """Panel title of one arm; the new arm says so when it is an older fit."""
    if arm == "new" and new is not None and getattr(new, "legacy_note", ""):
        return "new model, EARLIER-ROUND fit (re-expressed)"
    return ARM_LABEL[arm]


def upgrade_legacy_params(params: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    """An older parameter dict in the CURRENT parameterisation, or ``None``.

    Two steps between rounds are exact re-expressions, so a stored fit still
    describes the same process after them:

    * round 3's WHITE measurement noise of std ``sigma_w`` is the zero-memory
      limit of the measurement OU, so it becomes ``sigma_e = sigma_w`` at
      :data:`LEGACY_TAU_E_S` (lag-1 correlation 4.5e-5 on the 100 Hz grid);
    * round 4's per-rotor flight offsets ``s`` split into a common part and a
      per-rotor part, so they become ``s_c = 0`` and ``s_r = s``.

    Anything else returns ``None``: rounds 1 and 2 carried
    ``tau_fast``/``sigma_fast`` instead of the oscillator, which is a different
    model family and cannot be re-expressed.
    """
    if not {"f0", "zeta", "sigma_osc"} <= set(params):
        return None
    out = {k: v for k, v in params.items() if k != "corner_tau"}
    notes: list[str] = []
    if "sigma_w" in out and "tau_e" not in out:
        sigma_w = float(out.pop("sigma_w"))
        out["tau_e"] = LEGACY_TAU_E_S
        out["sigma_e"] = sigma_w
        notes.append(
            f"white sigma_w = {sigma_w:.4g} rev/s as the measurement OU at "
            f"tau_e = {LEGACY_TAU_E_S:g} s"
        )
    if "s" in out and "s_c" not in out:
        offsets = out.pop("s")
        out["s_c"] = 0.0
        out["s_r"] = offsets
        notes.append("per-rotor flight offsets s as s_c = 0 plus s_r = s")
    return (out, "; ".join(notes)) if notes else None


def load_new_fit(path: Path, legacy: bool = True) -> tuple[NewFit | None, str]:
    """``(fit, provenance)`` for one ``fits/new/<rig>.json``.

    A fit written by an earlier round carries a different parameter set. With
    ``legacy`` an exactly re-expressible fit goes through
    :func:`upgrade_legacy_params` and carries a ``legacy_note`` attribute,
    which every caption repeats; otherwise the fit is reported as unusable and
    the figures drop their new-model panel.
    """
    payload = json.loads(path.read_text())
    try:
        return NewFit.from_dict(payload), "current"
    except (KeyError, TypeError, ValueError) as exc:
        reason = f"not loadable by the current model ({exc!r})"
    upgraded = upgrade_legacy_params(payload.get("params") or {}) if legacy else None
    if upgraded is None:
        return None, reason
    params, how = upgraded
    fit = NewFit.from_dict({**payload, "params": params})
    # Provenance the current NewFit has no field for, read back through getattr
    # by _fit_band / _fit_nyquist / _missing_arm_note.
    fit.legacy_note = LEGACY_NOTE  # type: ignore[attr-defined]
    if "tau_e" not in (payload.get("params") or {}):
        fit.fit_rate_hz = float("nan")  # round 3 recorded no likelihood rate
    band = payload.get("band_hz")
    if band and len(tuple(band)) == 2:
        fit.band_hz = (float(band[0]), float(band[1]))  # type: ignore[attr-defined]
    return fit, f"earlier-round fit re-expressed exactly ({how}); {reason}"


def _fit_band(fit: NewFit) -> tuple[float, float] | None:
    """The band the fit was scored on, when the fit records one.

    Round 3 fitted a Whittle likelihood over a per-rig band and stored it as
    ``band_hz``; round 4 evaluates an exact time-domain likelihood and has no
    band at all.  Both are read here instead of asserting either.
    """
    band = getattr(fit, "band_hz", None)
    if band is None or len(tuple(band)) != 2:
        return None
    lo, hi = (float(band[0]), float(band[1]))
    return (lo, hi) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else None


def _fit_nyquist(fit: NewFit) -> float | None:
    """Half the rate the likelihood was evaluated at (round 4's
    ``fit_rate_hz``), or ``None`` when the fit does not record one."""
    rate = getattr(fit, "fit_rate_hz", None)
    try:
        rate = float(rate)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return 0.5 * rate if np.isfinite(rate) and rate > 0 else None


def _rotor_sigma(p: Params) -> float:
    """Mean per-rotor airborne std (rev/s), offsets excluded.

    Analytic when the parameters expose :meth:`Params.rotor_var`, else
    measured off a 60 s sample — the number is a plot coordinate, so a changed
    variance decomposition must not be able to break the figure.
    """
    try:
        return float(np.mean(np.sqrt(p.rotor_var(include_offset=False))))
    except (AttributeError, TypeError):
        x = p.sample_airborne(int(round(60.0 * RATE_HZ)), np.random.default_rng(0))
        return float(np.mean(np.std(x, axis=1)))


def _rotor_legend(fig: Figure, ax: Axes) -> None:
    """One figure-level rotor legend under the panels.

    A grid of small panels has no corner free of data, so the legend goes
    below the axes instead of over a trace.
    """
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles[:NUM_ROTORS],
        labels[:NUM_ROTORS],
        ncol=NUM_ROTORS,
        fontsize=8,
        loc="lower center",
        frameon=False,
    )


def _scale_axis(ax: Axes, values: list[float], axis: str, decades: float = 1.1) -> None:
    """Log scale only when the values really span ``decades``, and never with
    minor tick labels: a narrow log axis writes ``2 x 10^2`` over ``3 x 10^2``
    and becomes unreadable."""
    finite = np.asarray([v for v in values if np.isfinite(v) and v > 0], dtype=np.float64)
    target = ax.xaxis if axis == "x" else ax.yaxis
    if finite.size >= 2 and finite.max() / finite.min() >= 10.0**decades:
        ax.set_xscale("log") if axis == "x" else ax.set_yscale("log")
        target.set_minor_formatter(NullFormatter())
    target.set_tick_params(labelsize=8)


# ─── the fit's own block geometry, per mode (copy of the diag recipe) ─────────


def mode_periodogram(
    segments: list[np.ndarray], fs: float = RATE_HZ, block_s: float = BLOCK_S
) -> tuple[np.ndarray, np.ndarray, int]:
    """``(f, P, n_blocks)`` with ``P`` ``(4, F)`` one-sided MODE PSDs.

    The geometry the campaign's likelihood used in round 3 — Hann blocks of
    ``block_s`` at 50 % overlap, whole-segment blocks down to
    :data:`MIN_BLOCK_S`, block-mean detrend, taper-power corrected so
    ``sum_f P df`` is the block variance — expressed per mode
    (``m = M^T w / 4``).  Only the dominant
    (longest, most populated) block length is returned: the shorter
    whole-segment blocks live on a different frequency grid.
    """
    full = int(round(block_s * fs))
    hop = max(int(round(full * (1.0 - BLOCK_OVERLAP))), 1)
    min_len = int(round(MIN_BLOCK_S * fs))
    acc: dict[int, dict[str, Any]] = {}
    for seg in segments:
        seg = np.asarray(seg, dtype=np.float64)
        n = seg.shape[1]
        length = full if n >= full else n
        if length < min_len:
            continue
        entry = acc.setdefault(length, {"n": 0, "sum": None})
        w = hann(length, sym=False)
        for start in range(0, max(n - length + 1, 1), hop):
            block = seg[:, start : start + length]
            if block.shape[1] != length or not np.isfinite(block).all():
                continue
            centred = block - block.mean(axis=1, keepdims=True)
            spec = np.fft.rfft((MIXER.T @ centred) / NUM_ROTORS * w, axis=-1)
            power = (2.0 / (fs * float(np.sum(w**2)))) * np.abs(spec) ** 2
            entry["sum"] = power if entry["sum"] is None else entry["sum"] + power
            entry["n"] += 1
    usable = {k: v for k, v in acc.items() if v["n"]}
    if not usable:
        return np.zeros(0), np.zeros((NUM_ROTORS, 0)), 0
    length = max(usable, key=lambda k: (usable[k]["n"] * k, k))
    entry = usable[length]
    return np.fft.rfftfreq(length, d=1.0 / fs), entry["sum"] / entry["n"], int(entry["n"])


def model_mode_spectrum(p: Params, f: np.ndarray, fs: float = RATE_HZ) -> np.ndarray:
    """``(4, F)`` analytic spectrum of the model's MODE series ``M^T w / 4``.

    Read off :meth:`Params.spectral_matrix`, the model's ROTOR cross-spectrum:
    ``m = M^T w / 4`` is linear, so ``S_m = M^T S_w M / 16``.  Going through
    that method rather than assembling the components here is deliberate — it
    is the one spectral entry point that stays correct when the measurement
    term changes (round 3's white ``sigma_w``, round 4's per-rotor OU
    ``tau_e``/``sigma_e``).
    """
    s_w = p.spectral_matrix(np.atleast_1d(np.asarray(f, dtype=np.float64)), fs)
    s_m = np.einsum("ji,fjk,kl->fil", MIXER, s_w, MIXER) / float(NUM_ROTORS**2)
    return np.einsum("fii->if", s_m).real


def _model_segments(
    sampler: Any, lengths: list[int], seed: int, fs: float = RATE_HZ
) -> list[np.ndarray]:
    """One model segment per real segment length, capped at
    :data:`MODEL_SIM_CAP_S` simulated seconds (longest lengths first)."""
    order = sorted(lengths, reverse=True)
    cap = int(round(MODEL_SIM_CAP_S * fs))
    kept: list[int] = []
    total = 0
    for n in order:
        if total + n > cap and kept:
            break
        kept.append(n)
        total += n
    return [sampler(n, np.random.default_rng([seed, k])) for k, n in enumerate(kept)]


# ─── figures 1-2: the comparison grids (one ROW per rig, three COLUMNS) ───────

#: Airframe of each rig, for the row labels (`docs/data-catalog.md`).
RIG_VEHICLE = {
    "michaels": "DJI Matrice 100",
    "dregon": "MikroKopter quad",
    "neurobem_quad": "UZH RPG racing quad",
    "pitcn_quad": "NYU ARPL dragonfly17",
    "nanobench_cf21b": "Crazyflie 2.1 Brushless",
    "vid_m100": "DJI M100 + M3508",
    "blackbird_quad": "MIT Blackbird quad",
}

#: Column titles of both grids, in arm order.
COLUMN_TITLE = {
    "real": "real telemetry",
    "base": "current sampler, fitted",
    "new": "new model, fitted",
}


@dataclass
class Row:
    """One rig's row of a comparison grid: the same window from each arm."""

    rig: str
    series: dict[str, np.ndarray]
    fs: float
    ylim: tuple[float, float]
    xmax: float
    label: str
    caption: str
    meta: dict[str, Any]


def _row_label(rig: str, new: NewFit | None, extra: str = "") -> str:
    """Row label: rig, airframe, and any caveat this row carries."""
    notes = [extra] if extra else []
    if new is None:
        notes.append("no usable new fit")
    elif getattr(new, "legacy_note", ""):
        notes.append("new = earlier-round fit")
    tail = f"\n({'; '.join(notes)})" if notes else ""
    return f"{_rig_label(rig)}\n{RIG_VEHICLE.get(rig, 'quad')}{tail}"


def zoom_row(
    rig: str, flights: list[Flight], base: BaselineFit, new: NewFit | None, seed: int
) -> Row:
    """A 20 s window of each arm, from the middle of the longest real segment."""
    flight, start, n = _zoom_window(flights, ZOOM_S)
    series = {
        "real": np.asarray(flight.rps[:, start : start + n], dtype=np.float64),
        "base": base.sampler(flight.fs)(n, np.random.default_rng([seed, 1])),
    }
    if new is not None:
        series["new"] = new.sampler(flight.fs)(n, np.random.default_rng([seed, 2]))
    ylim = _finite_range(list(series.values()))
    return Row(
        rig=rig,
        series=series,
        fs=flight.fs,
        ylim=ylim,
        xmax=n / flight.fs,
        label=_row_label(rig, new),
        caption=(
            f"**{_rig_label(rig)}, {n / flight.fs:.0f} s zoom.** Real telemetry "
            f"from the middle of the longest airborne segment of "
            f"`{flight.flight}`, starting {start / flight.fs:.1f} s into the "
            f"recording. One sample of each fitted sampler follows it, on the "
            f"same 100 Hz grid and the same y-range "
            f"({ylim[0]:.0f}-{ylim[1]:.0f} rev/s). The colours are the mixer "
            f"rotor order." + _missing_arm_note(new)
        ),
        meta={
            "arms": list(series),
            "flight": flight.flight,
            "t0_s": round(start / flight.fs, 2),
            "window_s": round(n / flight.fs, 2),
        },
    )


def flight_row(
    rig: str, flights: list[Flight], base: BaselineFit, new: NewFit | None, seed: int
) -> Row:
    """A whole real recording against ``full_flight`` of each fit."""
    flight = _ground_flight(flights)
    real = np.asarray(flight.rps, dtype=np.float64)
    duration = float(real.shape[1] / flight.fs)
    model_s = max(duration, MIN_FLIGHT_S)
    series = {
        "real": real,
        "base": base.full_flight(model_s, flight.fs, np.random.default_rng([seed, 3])),
    }
    if new is not None:
        series["new"] = new.full_flight(model_s, flight.fs, np.random.default_rng([seed, 4]))
    idle = ground_level(flight)
    return Row(
        rig=rig,
        series=series,
        fs=flight.fs,
        ylim=_finite_range(list(series.values()), robust=True),
        xmax=max(duration, model_s),
        label=_row_label(rig, new, "" if idle is not None else "real: no ground phase"),
        caption=(
            f"**{_rig_label(rig)}, whole flight.** The complete recording "
            f"`{flight.flight}` lasts {duration:.0f} s. The frozen rule keeps "
            f"{_airborne_s(flight):.0f} s of it as airborne."
            + (
                f" The idle plateau sits at {idle:.0f} rev/s."
                if idle is not None
                else " This source starts and ends already airborne, so it shows no ground phase."
            )
            + f" Both fits then draw `full_flight({model_s:.0f} s)`: ground, "
            f"spin-up, warm-up idle, take-off ramp, the airborne process, "
            f"landing, spin-down. Only the airborne part is fitted. The phase "
            f"envelope is the shared scaffold `flight.wrap_airborne`. The "
            f"y-range covers the 0.2 to 99.8 percentile of all three cells, so "
            f"a rare logger glitch can fall outside the axes." + _missing_arm_note(new)
        ),
        meta={
            "arms": list(series),
            "flight": flight.flight,
            "duration_s": round(duration, 2),
            "model_duration_s": round(model_s, 2),
            "has_ground": idle is not None,
        },
    )


def _draw_row(axes: Any, row: Row, lw: float, titles: bool) -> None:
    """Fill three axes with one row: real, current sampler, new model."""
    for col, arm in enumerate(ARMS):
        ax = axes[col]
        x = row.series.get(arm)
        if x is None:
            ax.text(
                0.5,
                0.5,
                "no usable fit\nfor this arm",
                ha="center",
                va="center",
                fontsize=9,
                color="#b3261e",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            _rotor_lines(ax, np.arange(x.shape[1]) / row.fs, x, lw=lw)
            ax.set_ylim(*row.ylim)
            ax.set_xlim(0.0, row.xmax)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=7)
            if col:
                ax.set_yticklabels([])
        if titles:
            ax.set_title(COLUMN_TITLE[arm], fontsize=10, color=ARM_COLOR[arm])
    axes[0].set_ylabel(row.label, fontsize=8)


def fig_row(row: Row, out: Path, name: str) -> dict[str, Any]:
    """One rig's row as its own figure, for the per-rig tabs."""
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.0), squeeze=False)
    _draw_row(axes[0], row, lw=0.7, titles=True)
    for ax in axes[0]:
        ax.set_xlabel("time (s)", fontsize=8)
    _rotor_legend(fig, axes[0][0])
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    path = _save(fig, out, name)
    return {"path": path, "caption": row.caption, **row.meta}


def fig_grid(rows: list[Row], out: Path, name: str, kind: str) -> dict[str, Any]:
    """The comparison grid: one row per rig, three columns per row."""
    fig, axes = plt.subplots(len(rows), 3, figsize=(13.5, 2.6 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        _draw_row(axes[r], row, lw=0.55 if kind == "flight" else 0.7, titles=r == 0)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", fontsize=9)
    _rotor_legend(fig, axes[0][0])
    window = (
        f"{ZOOM_S:.0f} s airborne windows"
        if kind == "zoom"
        else "whole recordings and whole sampled flights"
    )
    fig.suptitle(f"{len(rows)} rigs, three arms, {window} — rev/s against time", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.985))
    path = _save(fig, out, name)
    legacy = [row.rig for row in rows if "earlier-round" in row.label]
    missing = [row.rig for row in rows if "new" not in row.series]
    caption = (
        f"**All {len(rows)} rigs at a glance"
        + (f", {ZOOM_S:.0f} s each.**" if kind == "zoom" else ", whole flights.**")
        + " Each row is one rig. The left cell is its real telemetry, the middle "
        "cell is the current sampler fitted to that rig, the right cell is the "
        "new model fitted to that rig. A row shares one y-range across its "
        "three cells, so the cells are directly comparable. Rotor colours are "
        "the mixer order in every cell. Row scales differ, because the rigs run "
        "from 78 to 278 rev/s."
    )
    if kind == "flight":
        caption += (
            " A row label says so when the real source carries no ground phase. "
            "Those recordings start and end already airborne."
        )
    if legacy:
        caption += (
            f" The right cell of {', '.join('`' + r + '`' for r in legacy)} comes "
            f"from an earlier round of the fit, re-expressed exactly in the "
            f"current parameterisation."
        )
    if missing:
        caption += (
            f" The right cell of {', '.join('`' + r + '`' for r in missing)} is "
            f"empty, because no fit there loads."
        )
    return {
        "path": path,
        "caption": caption,
        "rigs": [row.rig for row in rows],
        "legacy_rigs": legacy,
        "missing_rigs": missing,
    }


# ─── figure 3: the frozen statistics ──────────────────────────────────────────


def fig_stats(
    rig: str, st: dict[str, TrajStats], disc: dict[str, dict[str, float]], out: Path
) -> dict[str, Any]:
    verdict, strict = passes(disc["new"], disc["base"])
    fig = plt.figure(figsize=(12.0, 6.4))
    gs = fig.add_gridspec(2, 4, height_ratios=(1.0, 1.15), hspace=0.42, wspace=0.32)

    for i in range(NUM_ROTORS):
        ax = fig.add_subplot(gs[0, i])
        for arm in ARMS:
            ax.semilogx(
                LAGS_S,
                st[arm].acf[i],
                color=ARM_COLOR[arm],
                lw=1.6 if arm == "real" else 1.2,
                marker="o" if arm == "real" else None,
                ms=2.5,
                label=ARM_LABEL[arm] if i == 0 else None,
            )
        ax.axhline(0.0, color="#999999", lw=0.6)
        ax.set_ylim(-0.35, 1.05)
        ax.set_title(f"ACF {ROTOR_NAMES[i]}", fontsize=9)
        ax.set_xlabel("lag (s)", fontsize=8)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.set_ylabel("autocorrelation")
            ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

    for j, arm in enumerate(ARMS):
        ax = fig.add_subplot(gs[1, j])
        m = np.asarray(st[arm].xcorr, dtype=np.float64)
        im = ax.imshow(m, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
        ax.set_xticks(range(NUM_ROTORS), ROTOR_SHORT, fontsize=8)
        ax.set_yticks(range(NUM_ROTORS), ROTOR_SHORT, fontsize=8)
        ax.set_title(f"xcorr — {ARM_LABEL[arm]}", fontsize=9, color=ARM_COLOR[arm])
        for a in range(NUM_ROTORS):
            for b in range(NUM_ROTORS):
                val = m[a, b]
                ax.text(
                    b,
                    a,
                    "n/a" if not np.isfinite(val) else f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(val) > 0.6 else "black",
                )
        if j == 2:
            fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.9)

    ax = fig.add_subplot(gs[1, 3])
    x = np.arange(len(FAMILIES))
    vals = {
        arm: np.array([disc[arm][f] for f in FAMILIES], dtype=np.float64) for arm in ("base", "new")
    }
    for k, arm in enumerate(("base", "new")):
        ax.bar(
            x + (k - 0.5) * 0.38,
            vals[arm],
            width=0.36,
            color=ARM_COLOR[arm],
            label=ARM_LABEL[arm],
        )
    finite = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in vals.values()])
    if finite.size and finite.max() / max(finite.min(), 1e-12) > 50.0:
        ax.set_yscale("log")
    ax.set_xticks(x, [f.replace("_", "\n") for f in FAMILIES], fontsize=7)
    # Ticks and label on the right: the colourbar of the third heatmap sits
    # immediately left of this panel.
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("distance to real", fontsize=8)
    worse = [
        f for f in FAMILIES if vals["new"][FAMILIES.index(f)] > vals["base"][FAMILIES.index(f)]
    ]
    reason = f", {worse[0]} regresses" if (not verdict and worse) else ""
    ax.set_title(
        f"{'PASS' if verdict else 'FAIL'}: {sum(strict.values())}/{len(FAMILIES)} better{reason}",
        fontsize=9,
        color="#1a7f37" if verdict else "#b3261e",
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(f"{_rig_label(rig)} — the frozen statistics", fontsize=11)
    name = _save(fig, out, f"{rig}_stats.png")
    return {
        "path": name,
        "caption": (
            f"**{_rig_label(rig)}, the frozen statistics.** Top: the per-rotor "
            f"autocorrelation on the frozen log lag grid (0.05-10 s, 24 lags). "
            f"Bottom left to right: the 4x4 zero-lag cross-correlation of the real "
            f"telemetry and of both arms' samples, then the five discrepancy "
            f"families. The verdict shown is the frozen rule — no family may "
            f"regress and at least three must strictly improve — and reads "
            f"**{'PASS' if verdict else 'FAIL'}** here "
            f"({sum(strict.values())}/{len(FAMILIES)} families strictly better)."
        ),
        "verdict": bool(verdict),
        "strict": {k: bool(v) for k, v in strict.items()},
    }


# ─── figure 4: the mode spectra ───────────────────────────────────────────────


def fig_spectra(
    rig: str,
    flights: list[Flight],
    base: BaselineFit,
    new: NewFit | None,
    out: Path,
    seed: int,
) -> dict[str, Any]:
    fs = float((new.fs if new is not None else RATE_HZ) or RATE_HZ)
    real_segs = [np.asarray(f.rps[:, sl], dtype=np.float64) for f in flights for sl in _segments(f)]
    lengths = [seg.shape[1] for seg in real_segs]
    f_real, p_real, n_real = mode_periodogram(real_segs, fs)
    spectra = {"real": (f_real, p_real, n_real)}
    samplers = [("base", base.sampler(fs), 0)]
    if new is not None:
        samplers.append(("new", new.sampler(fs), 7))
    for arm, sampler, offset in samplers:
        spectra[arm] = mode_periodogram(_model_segments(sampler, lengths, seed + offset, fs), fs)
    arms = list(spectra)

    lo, hi = SPECTRA_BAND
    band = _fit_band(new) if new is not None else None
    nyquist = _fit_nyquist(new) if new is not None else None
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.8), sharex=True, sharey=True)
    for i, ax in enumerate(axes.ravel()):
        if band is not None:
            ax.axvspan(band[0], band[1], color="#f0f0f0", zorder=0)
        if nyquist is not None:
            ax.axvline(
                nyquist,
                color="#888888",
                lw=1.0,
                ls=":",
                label="likelihood Nyquist" if i == 0 else None,
            )
        for arm in arms:
            f, p, _ = spectra[arm]
            keep = (f >= lo) & (f <= hi) & np.isfinite(p[i]) & (p[i] > 0)
            if keep.any():
                ax.loglog(
                    f[keep],
                    p[i][keep],
                    color=ARM_COLOR[arm],
                    lw=1.5 if arm == "real" else 1.0,
                    alpha=1.0 if arm == "real" else 0.85,
                    label=_arm_label(arm, new) if i == 0 else None,
                )
        if new is not None and f_real.size:
            grid = np.geomspace(max(lo, f_real[1] if f_real.size > 1 else lo), hi, 300)
            analytic = model_mode_spectrum(new.params, grid, fs)[i]
            ax.loglog(
                grid,
                analytic,
                color=ARM_COLOR["new"],
                lw=1.0,
                ls="--",
                label="new model, analytic" if i == 0 else None,
            )
        ax.set_title(f"mode {i}: {MODE_NAMES[i]}", fontsize=10, color=MODE_COLORS[i])
        ax.set_xlim(lo, hi)
        ax.grid(alpha=0.25, which="both")
    for ax in axes[:, 0]:
        ax.set_ylabel("PSD ((rev/s)$^2$/Hz)")
    for ax in axes[1, :]:
        ax.set_xlabel("frequency (Hz)")
    axes[0, 0].legend(fontsize=8, loc="lower left", framealpha=0.9)
    scored = (
        f"shading: scored band {band[0]:g}-{band[1]:g} Hz"
        if band is not None
        else f"likelihood rate {2.0 * nyquist:g} Hz"
        if nyquist is not None
        else "full band"
    )
    fig.suptitle(
        f"{_rig_label(rig)} — mode spectra on 20 s Welch blocks ({scored}, {n_real} real blocks)",
        fontsize=11,
    )
    fig.tight_layout()
    name = _save(fig, out, f"{rig}_spectra.png")
    return {
        "path": name,
        "caption": (
            f"**{_rig_label(rig)}, mode spectra.** Welch spectra of the four "
            f"control modes (`M^T w / 4`) on the fit's block geometry: 20 s "
            f"Hann blocks at 50 % overlap, block-mean detrended, whole-segment "
            f"blocks down to 5 s. Real telemetry ({n_real} blocks) against samples "
            f"of the fits over the same segment lengths"
            + (", plus the new model's analytic mode spectrum (dashed)" if new else "")
            + f". {scored[0].upper() + scored[1:]}; the frozen statistics are "
            f"full-band regardless." + _missing_arm_note(new)
        ),
        "arms": arms,
        "n_real_blocks": int(n_real),
        "band_hz": list(band) if band is not None else None,
        "fit_nyquist_hz": nyquist,
    }


# ─── figures 5-7: the posterior ───────────────────────────────────────────────


def load_posterior(root: Path, fits: dict[str, NewFit]) -> tuple[Posterior | None, str]:
    """``(posterior, provenance)``: the stored rig posterior, else one fitted
    here from the loadable fits.

    ``posterior.json`` is written in the coordinates of the round that wrote
    it, so a stored posterior from an earlier round has the wrong dimension.
    Falling back to :func:`posterior.fit_posterior` over the fits that DO load
    keeps the figure produced by the campaign's own code rather than by a
    guessed coordinate migration — and the provenance string says which one
    the figure used.
    """
    path = root / "posterior.json"
    if path.is_file():
        try:
            return Posterior.from_json(path.read_text()), "stored `posterior.json`"
        except (KeyError, TypeError, ValueError) as exc:
            reason = f"stored `posterior.json` is unreadable by the current model ({exc!r})"
    else:
        reason = "no stored `posterior.json`"
    if not fits:
        return None, reason
    return (
        fit_posterior({rig: fit for rig, fit in fits.items()}),
        f"{reason}; refitted here from the {len(fits)} loadable fits",
    )


def load_draws(
    root: Path, post: Posterior, n_draws: int, seed: int, legacy: bool = True
) -> tuple[list[tuple[str, Params]], str]:
    """``(draws, provenance)``: the written draws, topped up from ``post``.

    A draw file written by an earlier round is re-expressed like a fit (see
    :func:`upgrade_legacy_params`) when ``legacy``; a file that is neither
    current nor re-expressible is skipped and replaced by a fresh draw, which
    the provenance string reports rather than hides.
    """
    files = sorted(
        (root / "draws").glob("*.json"),
        key=lambda p: int(re.sub(r"\D", "", p.stem) or "0"),
    )
    out: list[tuple[str, Params]] = []
    stale = 0
    upgraded = 0
    for path in files[:n_draws]:
        payload = json.loads(path.read_text())
        try:
            out.append((path.stem, Params.from_dict(payload)))
            continue
        except (KeyError, TypeError, ValueError):
            pass
        older = upgrade_legacy_params(payload) if legacy else None
        if older is None:
            stale += 1
            continue
        out.append((path.stem, Params.from_dict(older[0])))
        upgraded += 1
    n_stored = len(out)
    rng = np.random.default_rng(seed)
    while len(out) < n_draws:
        out.append((f"s{len(out)}", post.sample(rng)))
    parts = [f"{n_stored} of {n_draws} from `draws/*.json`"]
    if upgraded:
        parts.append(f"{upgraded} re-expressed from round 3")
    if stale:
        parts.append(f"{stale} unusable, skipped")
    if n_stored < n_draws:
        parts.append(f"{n_draws - n_stored} drawn here with seed {seed}")
    return out, ", ".join(parts)


#: Fit fields that are a rotor-speed LEVEL, so a posterior draw's own ``mu``
#: rescales them: the idle plateau ``full_flight`` ramps through, and the
#: sampler clamp a round may add (round 4's ``rps_min``/``rps_max``).
LEVEL_FIELDS = ("idle_rps", "rps_min", "rps_max")


def _level_ratios(fits: dict[str, NewFit]) -> dict[str, float]:
    """Mean fitted ``field / mean(mu)`` over the rigs, per level field present.

    A posterior draw carries airborne parameters only, while ``full_flight``
    needs an idle plateau (and, from round 4, the clamp the sampler applies).
    These ratios are measured off the rigs that ARE fitted, so a drawn drone
    gets its own ``mu`` times the corpus's idle/clamp geometry instead of an
    invented constant.
    """
    out: dict[str, float] = {}
    for field in LEVEL_FIELDS:
        ratios = []
        for fit in fits.values():
            value = np.mean(np.asarray(getattr(fit, field, np.nan), dtype=np.float64))
            mu = float(np.mean(fit.params.mu))
            if np.isfinite(value) and value > 0 and mu > 0:
                ratios.append(float(value) / mu)
        if ratios:
            out[field] = float(np.mean(ratios))
    return out


def _fit_from_draw(name: str, params: Params, template: NewFit, ratios: dict[str, float]) -> NewFit:
    """A :class:`NewFit` around one drawn parameter set.

    Built through ``to_dict``/``from_dict`` of an actually fitted rig so that
    every provenance field a round adds travels automatically; each level
    field becomes ``ratio * mu`` of the DRAW, keeping its per-rotor or scalar
    shape.  A level field the fits give no usable ratio for is NEUTRALISED
    rather than inherited — keeping a 200 rev/s rig's ceiling would clip a
    280 rev/s draw flat.
    """
    payload = dict(template.to_dict())
    payload["rig"] = f"draw_{name}"
    payload["params"] = params.to_dict()
    mu = np.asarray(params.mu, dtype=np.float64)
    for field in LEVEL_FIELDS:
        if field not in payload:
            continue
        ratio = ratios.get(field)
        if ratio is None:
            payload[field] = 0.0 if field.endswith("_min") else float("inf")
            continue
        payload[field] = (
            (ratio * mu).tolist()
            if np.ndim(payload[field]) > 0
            else float(ratio * float(np.mean(mu)))
        )
    return NewFit.from_dict(payload)


def fig_posterior_zoom(draws: list[tuple[str, Params]], out: Path, seed: int) -> dict[str, Any]:
    n = len(draws)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.0, 2.0 * nrow + 0.8), squeeze=False)
    n_samples = int(round(ZOOM_S * RATE_HZ))
    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.axis("off")
            continue
        name, params = draws[k]
        x = params.sample_airborne(n_samples, np.random.default_rng([seed, 11, k]))
        _rotor_lines(ax, np.arange(n_samples) / RATE_HZ, x, lw=0.6)
        # The title carries BOTH levels: a drawn flight offset can dwarf the
        # drawn operating point, and then "mean mu" alone misreads the panel.
        mu_mean = float(np.mean(params.mu))
        level = float(np.mean(x))
        title = (
            f"draw {name} — mu {mu_mean:.0f} rev/s"
            if abs(level - mu_mean) <= 0.05 * max(mu_mean, 1.0)
            else f"draw {name} — mu {mu_mean:.0f}, this flight {level:.0f} rev/s"
        )
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        if k % ncol == 0:
            ax.set_ylabel("rev/s", fontsize=8)
        if k >= n - ncol:
            ax.set_xlabel("time (s)", fontsize=8)
    _rotor_legend(fig, axes[0, 0])
    fig.suptitle(
        f"{n} random rigs from the posterior — {ZOOM_S:.0f} s airborne, own y-range each",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    name = _save(fig, out, "posterior_zoom_grid.png")
    return {
        "path": name,
        "caption": (
            f"**{n} random rigs, {ZOOM_S:.0f} s each.** One airborne sample per "
            f"posterior draw (`posterior.json` + `draws/*.json`), each panel on its "
            f"own y-range because the drawn operating points span "
            f"{min(np.mean(p.mu) for _, p in draws):.0f}-"
            f"{max(np.mean(p.mu) for _, p in draws):.0f} rev/s. Rotor colours are "
            f"the mixer order."
        ),
        "n_draws": n,
    }


def fig_posterior_flights(
    draws: list[tuple[str, Params]],
    template: NewFit,
    ratios: dict[str, float],
    duration_s: float,
    out: Path,
    seed: int,
) -> dict[str, Any]:
    use = draws[:N_FLIGHT_DRAWS]
    fig, axes = plt.subplots(len(use), 1, figsize=(11.0, 1.7 * len(use) + 0.9), squeeze=False)
    for k, (ax, (name, params)) in enumerate(zip(axes.ravel(), use, strict=True)):
        fit = _fit_from_draw(name, params, template, ratios)
        x = fit.full_flight(duration_s, RATE_HZ, np.random.default_rng([seed, 12, k]))
        _rotor_lines(ax, np.arange(x.shape[1]) / RATE_HZ, x, lw=0.6)
        ax.set_title(f"draw {name}", loc="left", fontsize=9)
        ax.set_ylabel("rev/s", fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0.0, duration_s)
    axes.ravel()[-1].set_xlabel("time (s)", fontsize=9)
    _rotor_legend(fig, axes.ravel()[0])
    scaled = ", ".join(f"{f} = {r:.2f} x mu" for f, r in sorted(ratios.items()))
    fig.suptitle(
        f"{len(use)} random rigs from the posterior — whole flights of "
        f"{duration_s:.0f} s ({scaled})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    name = _save(fig, out, "posterior_flight_grid.png")
    return {
        "path": name,
        "caption": (
            f"**{len(use)} random rigs, whole flights.** `full_flight("
            f"{duration_s:.0f} s)` per draw — the length is the median over rigs "
            f"of the rig's longest recording. A draw carries airborne parameters "
            f"only, so every level the whole-flight wrapper needs is the mean "
            f"fitted ratio over the fitted rigs times the draw's own `mu` "
            f"({scaled}); every other phase is the shared "
            f"`flight.wrap_airborne` scaffold."
        ),
        "duration_s": round(duration_s, 1),
        "level_ratios": {f: round(r, 4) for f, r in ratios.items()},
    }


def fig_posterior_planes(
    fits: dict[str, NewFit], draws: list[tuple[str, Params]], out: Path
) -> dict[str, Any]:
    def coords(p: Params) -> tuple[float, float, float, float, float, float]:
        mu = float(np.mean(p.mu))
        sigma = _rotor_sigma(p)
        return (
            mu,
            sigma / max(mu, 1e-9),
            float(p.f0[0]),
            float(p.zeta[0]),
            float(p.tau_slow[0]),
            float(p.sigma_slow[0]),
        )

    rig_pts = {rig: coords(fit.params) for rig, fit in fits.items()}
    draw_pts = [coords(p) for _, p in draws]
    planes = (
        (0, 1, "mean mu (rev/s)", "relative sigma (airborne std / mu)", "operating point"),
        (2, 3, "f0 of mode 0 (Hz)", "zeta of mode 0", "oscillator of the collective mode"),
        (
            4,
            5,
            "tau_slow of mode 0 (s)",
            "sigma_slow of mode 0 (rev/s)",
            "slow component of mode 0",
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3))
    for ax, (ix, iy, xl, yl, title) in zip(axes, planes, strict=True):
        ax.scatter(
            [d[ix] for d in draw_pts],
            [d[iy] for d in draw_pts],
            s=34,
            color="#9aa7b1",
            alpha=0.85,
            edgecolor="none",
            label="posterior draws",
        )
        for rig, pt in rig_pts.items():
            ax.scatter(pt[ix], pt[iy], s=46, color="#b3261e", marker="D", zorder=3)
            ax.annotate(
                _rig_label(rig),
                (pt[ix], pt[iy]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7.5,
            )
        every = [*(d[ix] for d in draw_pts), *(p[ix] for p in rig_pts.values())]
        _scale_axis(ax, every, "x")
        _scale_axis(ax, [*(d[iy] for d in draw_pts), *(p[iy] for p in rig_pts.values())], "y")
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, which="both")
    axes[0].scatter([], [], s=46, color="#b3261e", marker="D", label="fitted rigs")
    axes[0].legend(fontsize=8, loc="best")
    plural = "rigs" if len(rig_pts) != 1 else "rig"
    fig.suptitle(f"The {len(rig_pts)} fitted {plural} and the posterior draws", fontsize=11)
    fig.tight_layout()
    name = _save(fig, out, "posterior_planes.png")
    return {
        "path": name,
        "caption": (
            f"**Where the draws sit relative to the rigs.** Three planes of the "
            f"rig vector: the operating point against the relative airborne "
            f"spread, the collective mode's oscillator (`f0`, `zeta`), and its "
            f"slow component (`tau_slow`, `sigma_slow`). Red diamonds are the "
            f"{len(rig_pts)} fitted rigs, grey dots the {len(draw_pts)} draws. "
            f"The posterior is a diagonal Gaussian in scale-free coordinates "
            f"with a floored per-coordinate width, so the draws must spread "
            f"wider than the rigs — with {len(rig_pts)} drones the raw spread "
            f"would be far too tight to sample from."
        ),
        "n_rigs": len(rig_pts),
        "n_draws": len(draw_pts),
    }


# ─── exploration PNGs ─────────────────────────────────────────────────────────


def copy_explore(root: Path, out: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for name, caption in EXPLORE_FIGS:
        src = root / "explore" / name
        if not src.is_file():
            continue
        dst = out / f"explore_{name}"
        shutil.copyfile(src, dst)
        entries.append(
            {"path": dst.name, "caption": caption, "source": str(src.relative_to(_ROOT))}
        )
    return entries


# ─── main ─────────────────────────────────────────────────────────────────────


def _latest_round(root: Path) -> str:
    rounds = sorted(
        (root / "rounds").glob("round*.json"),
        key=lambda p: int(re.sub(r"\D", "", p.stem) or "0"),
    )
    return rounds[-1].stem if rounds else "unversioned"


def _typical_flight_s(root: Path, rigs: list[str]) -> float:
    """Length of the whole-flight draws: the median over rigs of the rig's
    LONGEST recording.

    Not the median recording: neurobem contributes 247 dropout-free segments
    with a 7 s median, which would collapse the figure to a take-off ramp.
    The longest recording per rig is what "a whole flight of this rig" means.
    """
    longest: list[float] = []
    for rig in rigs:
        path = root / "stats" / "real" / f"{rig}.flights.json"
        if not path.is_file():
            continue
        durations = [float(f["duration_s"]) for f in json.loads(path.read_text())["flights"]]
        if durations:
            longest.append(max(durations))
    return max(float(np.median(longest)) if longest else MIN_FLIGHT_S, MIN_FLIGHT_S)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--rigs", default=",".join(RIGS), help="comma-separated rig ids")
    ap.add_argument("--results", default="results/rps_traj", help="campaign results root")
    ap.add_argument(
        "--out",
        default="docs/explainers/rps-trajectory-model",
        help="figure folder (PNGs + figure_index.json)",
    )
    ap.add_argument("--round", default=None, help="round label for the index (default: latest)")
    ap.add_argument("--n-draws", type=int, default=12, help="posterior draws in the zoom grid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--legacy-fits",
        choices=("upgrade", "skip"),
        default="upgrade",
        help=(
            "what to do with a fits/new/<rig>.json written by an earlier round: "
            "re-express a round-3 fit in the current parameterisation and label "
            "every figure that uses it (default), or drop its panel entirely"
        ),
    )
    args = ap.parse_args()

    root = (_ROOT / args.results).resolve()
    out = (_ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rigs = [r for r in (s.strip() for s in args.rigs.split(",")) if r]
    label = args.round or _latest_round(root)

    index: dict[str, Any] = {
        "round": label,
        "rate_hz": RATE_HZ,
        "seed": args.seed,
        "results_root": str(root.relative_to(_ROOT)),
        "per_rig_order": ["zoom", "flight", "stats", "spectra"],
        "rigs": [],
        "figures": {},
        "posterior": {},
        "grids": {},
        "skipped": {},
        "fit_provenance": {},
        "fit_facts": {},
        "explore": copy_explore(root, out),
    }

    fits: dict[str, NewFit] = {}
    zoom_rows: list[Row] = []
    flight_rows: list[Row] = []
    for rig in rigs:
        base_path = root / "fits" / "base" / f"{rig}.json"
        new_path = root / "fits" / "new" / f"{rig}.json"
        if not base_path.is_file():
            reason = "no baseline fit under fits/base"
            index["skipped"][rig] = reason
            print(f"[skip] {rig}: {reason}", flush=True)
            continue
        base = BaselineFit.from_json(base_path.read_text())
        new, provenance = (
            load_new_fit(new_path, legacy=args.legacy_fits == "upgrade")
            if new_path.is_file()
            else (None, "no fit under fits/new")
        )
        if new is None:
            # A results tree from an earlier round is a normal state between
            # rounds. Report it per rig, draw the arms that exist, and let the
            # page say which panel is missing.
            index["skipped"][rig] = provenance
            print(f"[partial] {rig}: {provenance}", flush=True)
        else:
            fits[rig] = new
        index["fit_provenance"][rig] = provenance
        if new is not None:
            # Provenance straight from the fit artefact: summary.json can be a
            # round behind it while a round is landing rig by rig, and the fit
            # is the file that carries the likelihood's own bookkeeping.
            index["fit_facts"][rig] = {
                key: value
                for key, value in new.to_dict().items()
                if key not in ("params", "rig", "kind") and np.ndim(value) == 0
            }
            index["fit_facts"][rig]["legacy"] = bool(getattr(new, "legacy_note", ""))
        flights = load_rig(rig)
        zr = zoom_row(rig, flights, base, new, args.seed)
        fr = flight_row(rig, flights, base, new, args.seed)
        zoom_rows.append(zr)
        flight_rows.append(fr)
        entries = {
            "zoom": fig_row(zr, out, f"{rig}_zoom.png"),
            "flight": fig_row(fr, out, f"{rig}_flight.png"),
            "spectra": fig_spectra(rig, flights, base, new, out, args.seed),
        }
        stat_paths = {arm: root / "stats" / arm / f"{rig}.json" for arm in ARMS}
        if all(p.is_file() for p in stat_paths.values()):
            st = {arm: TrajStats.from_json(p.read_text()) for arm, p in stat_paths.items()}
            disc = {arm: discrepancy(st[arm], st["real"]) for arm in ("base", "new")}
            entries["stats"] = fig_stats(rig, st, disc, out)
        else:
            print(f"[warn] {rig}: no stored statistics, skipping the stats figure", flush=True)
        index["rigs"].append(rig)
        index["figures"][rig] = {
            key: entries[key] for key in index["per_rig_order"] if key in entries
        }
        print(f"[done] {rig}: {', '.join(e['path'] for e in entries.values())}", flush=True)

    if zoom_rows:
        index["grids"] = {
            "zoom": fig_grid(zoom_rows, out, "grid_zoom.png", "zoom"),
            "flight": fig_grid(flight_rows, out, "grid_flight.png", "flight"),
        }
        index["grid_order"] = ["zoom", "flight"]
        print("[done] comparison grids: grid_zoom.png, grid_flight.png", flush=True)

    post, post_provenance = load_posterior(root, fits)
    index["posterior_source"] = post_provenance
    if post is not None and fits:
        draws, draw_provenance = load_draws(
            root,
            post,
            max(args.n_draws, N_FLIGHT_DRAWS),
            args.seed,
            legacy=args.legacy_fits == "upgrade",
        )
        ratios = _level_ratios(fits)
        # Provenance template: the rig of median operating point, so a drawn
        # drone inherits real fit bookkeeping and only its levels are rescaled.
        template = sorted(fits.values(), key=lambda f: float(np.mean(f.params.mu)))[len(fits) // 2]
        posterior = {
            "zoom_grid": fig_posterior_zoom(draws, out, args.seed),
            "flight_grid": fig_posterior_flights(
                draws, template, ratios, _typical_flight_s(root, rigs), out, args.seed
            ),
            "planes": fig_posterior_planes(fits, draws, out),
        }
        index["posterior"] = posterior
        index["posterior_order"] = list(posterior)
        index["posterior_rigs"] = list(post.rigs)
        index["posterior_template_rig"] = template.rig
        index["draws_source"] = draw_provenance
        print(f"[done] posterior figures ({post_provenance}; draws: {draw_provenance})", flush=True)
    else:
        print(f"[skip] posterior figures: {post_provenance}", flush=True)

    index["n_figures"] = (
        sum(len(v) for v in index["figures"].values())
        + len(index.get("grids", {}))
        + len(index.get("posterior", {}))
        + len(index["explore"])
    )
    (out / "figure_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"[index] {out / 'figure_index.json'}: {index['n_figures']} figures, round {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
