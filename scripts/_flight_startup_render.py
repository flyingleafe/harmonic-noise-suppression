"""REAL vs the ASSEMBLED FLIGHT MODEL: whole flights and the startup close-up.

A research-sandbox probe of ``experiments.stochastic_fit.flight_model``: the
pre-revision (stage-2) fits are per-operating-point, so a whole flight is
rendered by blending them along the instantaneous rotor speed, with exact
silence at zero speed. Nothing here is a revised-phase campaign candidate and
nothing here makes a gate claim.

Conventions follow ``scripts/_revised_ab_render.py`` (§16.5/§19.5 of
``docs/explainers/rotor-noise-fit.qmd``): the same real clip, the RAW rotor
trajectory, one seed per rig, one common playback gain per window, and an index
JSON carrying the provenance. The one deliberate difference: a flight IS a level
ramp, so NOTHING is per-window normalised. The model render keeps its own
absolute level and gets ONE global scalar, chosen so its RMS over a stated
STEADY segment equals the real clip's RMS over that same segment — the ramp
that leads up to it is then the model's own prediction. That segment is the
last 1.0 s for the 6 s startup windows and the last 5.0 s for the whole
flights.

Anchors (pre-revision fits, with the mean rotor speed of their own support
MEASURED from telemetry in this script, never hard-coded):

* michaels: ``results/S2/standby.json`` (FLY125 [2, 10) s) and
  ``results/S2/cruise_8clip.json`` (FLY125 [32, 48) s);
* dregon: ``results/S2/dregon_room2_cruise_refined.json``
  (``free-flight_nosource_room2`` [1512727417.205, +16) s).

Windows (DREGON ``start_s`` is absolute epoch seconds):

* startup — FLY125 [0, 6) s; ``free-flight_nosource_room2`` [1512727390.5, +6);
* flight — FLY125 [0, 90) s; ``free-flight_nosource_room2``
  [1512727390.5, +70), which stays inside the telemetry span that ends at
  1512727463.994.

FLY125's flight window is 90 s and not the 120 s the assignment asked for. The
reason is PEAK MEMORY, not CPU: one anchor render measures 3.26 GB peak RSS at
30 s and 5.45 GB at 60 s (``stochastic_rotor_noise.synthesize`` holds several
``(mics, frames, 32769)`` float64 spectra at a 65536-sample FFT), which
projects to about 9.8 GB at 120 s against 12 GB available on this machine. 90 s
projects to 7.7 GB and measured 34.2 s of CPU per anchor at 60 s, so the whole
run is a few minutes. Nothing structural is lost: FLY125 is at rest until 2 s,
in standby near 35.5 rev/s from 2 to 16 s, ramps from 16 to 20 s and holds
cruise near 80 rev/s from 20 s onwards, so [0, 90) already spans every one of
the model's blend regions and 90-120 s would add only more cruise.

    PYTHONPATH="$PWD/src:$PWD/scripts" python scripts/_flight_startup_render.py

Writes, under ``docs/explainers/flight-startup/``:

* ``startup_<rig>.png`` (real spectrogram, model spectrogram on the real
  panel's colour scale, level ramp + speed) and ``flight_<rig>.png`` (the same
  two spectrograms, the level ramp, and the speed with the blend weights);
* ``{startup,flight}_<rig>_{real,model}.wav`` — mic 0, 16 kHz, one shared
  playback gain per window;
* ``{startup,flight}_<rig>.json`` — metrics and provenance;
* ``blend_weights.png`` — the weight functions against SPEED, both rigs;
* ``verify.json`` — the mechanism checks: exact zero-speed silence, anchor
  phase lock (same seed against a different seed), and the blend evaluated at
  an anchor's own speed against that anchor's own render, plus the measured
  justification for any floor-exponent rebase;
* ``low_order_diagnosis.json`` — the three-curve low-order diagnosis (real /
  the fit's own forward law / the render) on each anchor's own fit support,
  plus a fourth curve with the coherent-incoherent split done consistently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from experiments.stochastic_fit import clips as C
from experiments.stochastic_fit import data as C_DATA
from experiments.stochastic_fit import flight_model as FM
from experiments.stochastic_fit import stage2 as S2

OUT = Path("docs/explainers/flight-startup")
SEED = 4242
#: The level-ramp resolution: 50 ms frames, the same window the blend's speed
#: is smoothed over.
RAMP_S = 0.05
#: An onset is where a 50 ms level first stands this far above the arm's own
#: first-100 ms level.
ONSET_DB = 20.0
#: A single WAV larger than this also gets a cruise excerpt for the player.
WAV_MAX_MB = 10.0
EXCERPT_S = 12.0
COLOURS = {"real": "#111111", "model": "#1f77b4"}
ANCHOR_COLOURS = ("#2ca02c", "#d62728", "#9467bd")
SPEED_COLOUR = "#ff7f0e"
RAW_KEY = {"michaels": "rps", "dregon": "motors_command"}
DATASET = {"michaels": "michaels-frames", "dregon": "DREGON-frames"}
RIGS = ("michaels", "dregon")

#: (fit path, clip selector, label, support recording, support start_s, seconds)
ANCHORS: dict[str, list[tuple[Path, str, str, str, float, float]]] = {
    "michaels": [
        (Path("results/S2/standby.json"), "fly125_cruise_00", "standby", "FLY125", 2.0, 8.0),
        (Path("results/S2/cruise_8clip.json"), "fly125_cruise_00", "cruise", "FLY125", 32.0, 16.0),
    ],
    "dregon": [
        (
            Path("results/S2/dregon_room2_cruise_refined.json"),
            "free-flight_nosource_room2_cruise_00",
            "cruise",
            "free-flight_nosource_room2",
            1512727417.2050455,
            16.0,
        )
    ],
}
#: (recording, start_s, seconds, match_s) per rig per window kind. ``match_s``
#: is the trailing STEADY segment the one global scalar is fitted on.
WINDOWS: dict[str, dict[str, tuple[str, float, float, float]]] = {
    "startup": {
        "michaels": ("FLY125", 0.0, 6.0, 1.0),
        "dregon": ("free-flight_nosource_room2", 1512727390.5, 6.0, 1.0),
    },
    "flight": {
        "michaels": ("FLY125", 0.0, 90.0, 5.0),
        "dregon": ("free-flight_nosource_room2", 1512727390.5, 70.0, 5.0),
    },
}
#: Steady segments of each flight window, judged separately: a whole-window
#: number would average the regimes the assembly is supposed to distinguish.
#: The LAST one is the SETTLED segment the spectral shape is read on.
FLIGHT_SEGMENTS: dict[str, list[tuple[str, float, float]]] = {
    "michaels": [("standby", 4.0, 15.0), ("cruise", 25.0, 85.0)],
    "dregon": [("cruise", 6.0, 65.0)],
}
#: Where the startup close-up tracks the strongest comb line: (t0, t1, step,
#: f_lo, f_hi) and the optional (peak window, trough window) of the rig's own
#: speed transient. 1.5-4 kHz is where both rigs' combs are visible by eye, but
#: the two rigs reach their settled speed at different times, so the tracking
#: window is per rig rather than shared.
COMB_TRACK: dict[str, tuple[tuple[float, float, float, float, float], Any]] = {
    "michaels": ((1.0, 4.0, 0.1, 1500.0, 4000.0), ((1.4, 1.7), (2.0, 3.0))),
    "dregon": ((2.6, 6.0, 0.1, 1500.0, 4000.0), None),
}
#: Edges of the ABSOLUTE band diagnostic. Reaches down to 20 Hz on purpose:
#: ``accept_stats.ltas_bands`` is shape-only and starts at 100 Hz, so it is
#: blind to a low-frequency absolute-level excess.
ABSOLUTE_BAND_EDGES = (20.0, 50.0, 100.0, 200.0, 300.0, 700.0, 1500.0, 3000.0, 5000.0, 7900.0)
#: Orders probed on and between the comb, to tell a LINE level error from a
#: FLOOR level error where the absolute band residual lives.
LOW_ORDER_PROBE = (1, 2, 3, 4, 6, 10)
#: The three-curve low-order diagnosis (section 4.3 of the explainer) runs on
#: each anchor's OWN fit support, so nothing is attributable to the blend:
#: which anchor, and its support window (recording, start_s, seconds).
DIAGNOSIS_SUPPORT: dict[str, tuple[str, str, float, float]] = {
    "michaels": ("cruise", "FLY125", 32.0, 16.0),
    "dregon": ("cruise", "free-flight_nosource_room2", 1512727417.2050455, 16.0),
}
#: Orders and broadband bands the three curves are integrated over. The order
#: bands are +-0.35 of the line spacing, i.e. the whole order with no gap and
#: no overlap, so a line that has moved between rotors is still counted once.
DIAGNOSIS_ORDERS = (1, 2, 3, 4, 5, 6)
DIAGNOSIS_BANDS = ((300, 700), (700, 1500), (1500, 3200), (3200, 4500), (4500, 7900))
#: Analysis grid of the diagnosis: the fit's OWN front end.
DIAGNOSIS_N_FFT = 16384


def spec_panel(
    ax, x: np.ndarray, title: str, vlim: tuple[float, float] | None = None
) -> tuple[float, float]:
    """One spectrogram panel; returns the colour limits it used.

    ``vlim`` lets the model panel inherit the REAL panel's limits, so the two
    are visually comparable instead of each being auto-scaled to its own peak.
    """
    n, hop = 2048, 512
    w = np.hanning(n + 1)[:n]
    fr = np.stack([x[s : s + n] * w for s in range(0, x.size - n, hop)])
    S = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=-1)).T + 1e-9)
    if vlim is None:
        top = float(np.percentile(S, 99.5))
        vlim = (top - 75.0, top)
    ax.imshow(
        S,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, x.size / S2.SR, 0, S2.SR / 2000],
        vmin=vlim[0],
        vmax=vlim[1],
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("s")
    return vlim


def rms(x: np.ndarray) -> float:
    """Whole-array RMS, exactly ``stochastic_rotor_noise.synthesize``'s definition."""
    return float(np.sqrt(np.mean(np.square(x))))


def dbfs(x: np.ndarray) -> float:
    return float(20.0 * np.log10(max(rms(x), 1e-12)))


def load_real(rig: str, recording: str, start_s: float, seconds: float, clip_id: str):
    return C.decimate(
        C.load_clip(
            DATASET[rig],
            recording,
            start_s,
            seconds,
            rps_key=RAW_KEY[rig],
            clip_id=clip_id,
        ),
        S2.SR,
    )


def prev_entry(fit_path: Path, selector: Any) -> dict[str, Any]:
    """One clip's ENTRY of a previous-model fit — ``params`` AND ``scores``.

    The entry, not the parameter dict: ``scores.power_scale`` is the fit's own
    periodogram normalisation and the levels in ``params`` are expressed in it,
    so :func:`flight_model.physical_export` needs both halves.
    """
    summary = json.loads(fit_path.read_text())
    entries = list(summary["clips"].values())
    if isinstance(selector, int):
        return entries[selector]
    for cid, entry in summary["clips"].items():
        if cid.startswith(str(selector)):
            return entry
    raise SystemExit(f"{fit_path}: no clip starts with {selector!r}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ramp_db(x: np.ndarray) -> np.ndarray:
    """Per-50 ms whole-array RMS in dB; the level ramp of one arm."""
    n = int(round(RAMP_S * S2.SR))
    k = x.shape[-1] // n
    fr = x[..., : k * n].reshape(*x.shape[:-1], k, n)
    return 10.0 * np.log10(np.mean(np.square(fr), axis=(*range(x.ndim - 1), -1)) + 1e-24)


def onset_s(x: np.ndarray, level: np.ndarray) -> tuple[float | None, float]:
    """``(onset, base_db)``: the first 50 ms frame centre whose level stands
    ``ONSET_DB`` above the arm's OWN first-100 ms level, and that reference.

    The reference is returned because it is not comparable between arms: the
    model is EXACTLY silent while the rotors are stopped, so its base is the
    ``1e-12`` guard (-240 dB) and the threshold is crossed by the first sample
    that is not zero. The real arm's base is a genuine recorder/ambient floor.
    An onset is therefore a statement about one arm's own dynamic range, not a
    like-for-like arrival time, unless both bases are real floors.
    """
    base = 20.0 * np.log10(max(rms(x[..., : int(0.1 * S2.SR)]), 1e-12))
    hit = np.flatnonzero(level >= base + ONSET_DB)
    return (None if hit.size == 0 else float((hit[0] + 0.5) * RAMP_S)), base


_MODELS: dict[str, tuple[FM.FlightModel, list[dict[str, Any]]]] = {}


def build_model(rig: str) -> tuple[FM.FlightModel, list[dict[str, Any]]]:
    """Assemble one rig's flight model, measuring anchor speeds from telemetry.

    The provenance rows carry each anchor's FITTED speed exponents as they
    stand in the file — ``amp_exp`` for the comb, ``floor_exp`` for the
    broadband floor — whether :func:`flight_model.assemble` had to rebase the
    floor one because it is negative and therefore unbounded at zero speed,
    and the LEVEL coordinate: the clip's own ``scores.power_scale`` and the two
    declared offsets :func:`flight_model.physical_export` folds in. The two
    rigs need not agree on any of it, so the comparison must disclose it.
    """
    if rig in _MODELS:
        return _MODELS[rig]
    anchors: list[FM.Anchor] = []
    provenance: list[dict[str, Any]] = []
    for path, selector, label, recording, start_s, seconds in ANCHORS[rig]:
        support = load_real(rig, recording, start_s, seconds, f"anchor_{rig}_{label}")
        speed = float(np.asarray(support.rps, dtype=np.float64).mean())
        entry = prev_entry(path, selector)
        anchor = FM.anchor_from_entry(entry, speed_rps=speed, label=label)
        anchors.append(anchor)
        raw = entry["params"]
        amp_exp = float(raw.get("amp_exp", 0.0))
        provenance.append(
            dict(
                label=label,
                speed_rps=round(speed, 4),
                fit=str(path),
                selector=selector,
                support=f"{recording}@{start_s:.3f}+{seconds:g}",
                amp_exp_fitted=round(amp_exp, 4),
                floor_exp_fitted=round(float(raw.get("floor_exp", amp_exp)), 4),
                power_scale=float((entry.get("scores") or {}).get("power_scale", 1.0)),
                power_scale_folded_db=round(float(anchor.export["power_scale_folded_db"]), 4),
                render_units_rate_factor_db=round(
                    float(anchor.export["render_units_rate_factor_db"]), 4
                ),
                render_units_sample_rate_work=int(anchor.export["render_units_sample_rate_work"]),
                sha256=sha256(path),
            )
        )
    order = np.argsort([a.speed_rps for a in anchors])
    model = FM.assemble(rig, [anchors[i] for i in order])
    provenance = [provenance[i] for i in order]
    for row, anchor in zip(provenance, model.anchors, strict=True):
        row["floor_exp_rebased"] = any("floor_exp" in n for n in anchor.notes)
        row["floor_exp_used"] = round(float(anchor.export.get("floor_exp", 0.0)), 4)
    _MODELS[rig] = (model, provenance)
    return _MODELS[rig]


def silence_check(model: FM.FlightModel, n_mics: int) -> dict[str, Any]:
    """Render 1.0 s of stopped rotors and demand EXACT zero."""
    zeros = np.zeros((model.n_rotors, S2.SR), dtype=np.float64)
    r = FM.render_flight(model, zeros, n_mics=n_mics, seed=SEED)
    peak = float(np.max(np.abs(r.audio)))
    return dict(
        exact=bool(peak == 0.0),
        peak_abs=peak,
        max_weight=float(np.max(r.weights)),
        max_speed_rps=float(np.max(r.speed_rps)),
    )


def compare(rig: str, kind: str) -> dict[str, Any]:
    """Render one window of one rig and measure it against the real clip.

    NO gain is applied to the model. Since the anchors go through
    :func:`flight_model.physical_export` and are rendered with
    ``normalize_rms=None``, the model's output is already in the real
    recording's own units: ``power_scale`` is the normalisation of THAT
    recording's periodogram, so folding it back puts the parameters in the
    recorder's digital scale. The level comparison is therefore direct, and
    the gain that WOULD be needed to match the trailing steady segment is kept
    only as a diagnostic (``match_gain_db_would_be``) — it is the residual of
    an absolute-level prediction, not a normalisation.
    """
    recording, start_s, seconds, match_s = WINDOWS[kind][rig]
    model, provenance = build_model(rig)
    real = load_real(rig, recording, start_s, seconds, f"{kind}_{recording}_{start_s:.3f}")
    xr = np.asarray(real.audio, dtype=np.float64)
    rps = np.asarray(real.rps, dtype=np.float64)
    n_mics = int(xr.shape[0])

    render = FM.render_flight(model, rps, n_mics=n_mics, seed=SEED)
    xm = render.audio
    n = min(xr.shape[-1], xm.shape[-1])
    xr, xm = xr[:, :n], xm[:, :n]

    tail = int(round(match_s * S2.SR))
    gain = rms(xr[:, -tail:]) / max(rms(xm[:, -tail:]), 1e-30)

    waves = {"real": xr, "model": xm}
    levels = {k: ramp_db(v) for k, v in waves.items()}
    measured = {k: onset_s(waves[k], levels[k]) for k in waves}
    frame = int(round(RAMP_S * S2.SR))
    nf = levels["real"].size
    speed_ramp = render.speed_rps[:n][: nf * frame].reshape(nf, frame).mean(axis=1)
    weight_ramp = render.weights[:, :n][:, : nf * frame].reshape(-1, nf, frame).mean(axis=2)
    return dict(
        rig=rig,
        kind=kind,
        recording=recording,
        start_s=start_s,
        seconds=seconds,
        match_s=match_s,
        model=model,
        provenance=provenance,
        render=render,
        n_mics=n_mics,
        waves=waves,
        levels=levels,
        onsets={k: v[0] for k, v in measured.items()},
        onset_base={k: round(v[1], 2) for k, v in measured.items()},
        corr=float(np.corrcoef(levels["real"], levels["model"])[0, 1]),
        gain=float(gain),
        t_ramp=(np.arange(nf) + 0.5) * RAMP_S,
        speed_ramp=speed_ramp,
        weight_ramp=weight_ramp,
    )


def level_panel(ax, ctx: dict[str, Any]) -> None:
    """Per-50 ms level of both arms, with the onset marks."""
    for name in ctx["waves"]:
        ax.plot(
            ctx["t_ramp"],
            ctx["levels"][name],
            lw=1.2,
            color=COLOURS[name],
            label=f"{name} {RAMP_S * 1000:.0f} ms RMS",
        )
    for name, t in ctx["onsets"].items():
        if t is not None:
            ax.axvline(t, color=COLOURS[name], ls=":", lw=1.0)
    ax.set_xlabel("s")
    ax.set_ylabel("dB, model at its OWN physical level (no gain)")
    # The model is EXACTLY silent while the rotors are stopped, so its first
    # frames sit at the -240 dB log guard. Show the region both arms occupy
    # instead of letting that crush the axis.
    real, mod = ctx["levels"]["real"], ctx["levels"]["model"]
    floor = min(float(real.min()), float(mod.max()) - 60.0) - 8.0
    ax.set_ylim(floor, max(float(real.max()), float(mod.max())) + 4.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    shown = {k: "n/a" if v is None else f"{v:.2f} s" for k, v in ctx["onsets"].items()}
    ax.set_title(
        f"level ramp — corr {ctx['corr']:.3f}, onset real {shown['real']} / model {shown['model']}",
        fontsize=9,
    )


def suptitle(ctx: dict[str, Any]) -> str:
    return (
        f"{ctx['rig']}: anchors "
        + ", ".join(
            f"{p['label']} {p['speed_rps']:.1f} rev/s "
            f"(amp_exp {p['amp_exp_fitted']:.2f}, floor_exp {p['floor_exp_fitted']:.2f}"
            + (" REBASED to 0)" if p["floor_exp_rebased"] else ")")
            for p in ctx["provenance"]
        )
        + f" — K={ctx['model'].n_orders}, seed {SEED}, "
        f"physical level residual over last {ctx['match_s']:g} s: "
        f"{-20 * np.log10(ctx['gain']):+.2f} dB (no gain applied)"
    )


def startup_figure(ctx: dict[str, Any]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    rec = ctx["recording"]
    vlim = spec_panel(axes[0], ctx["waves"]["real"][0], f"{rec} startup — real (mic 0)")
    spec_panel(axes[1], ctx["waves"]["model"][0], f"{rec} startup — flight model (mic 0)", vlim)
    axes[0].set_ylabel("kHz")
    level_panel(axes[2], ctx)
    ax2 = axes[2].twinx()
    ax2.plot(ctx["t_ramp"], ctx["speed_ramp"], lw=1.2, color=SPEED_COLOUR, ls="--")
    ax2.set_ylabel("mean rotor speed (rev/s)", color=SPEED_COLOUR)
    ax2.tick_params(axis="y", labelcolor=SPEED_COLOUR)
    fig.suptitle(suptitle(ctx), fontsize=9)
    fig.tight_layout()
    path = OUT / f"startup_{ctx['rig']}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def flight_figure(ctx: dict[str, Any]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 8.4), gridspec_kw={"height_ratios": [1.15, 1]})
    rec = ctx["recording"]
    vlim = spec_panel(axes[0, 0], ctx["waves"]["real"][0], f"{rec} flight — real (mic 0)")
    spec_panel(axes[0, 1], ctx["waves"]["model"][0], f"{rec} flight — flight model (mic 0)", vlim)
    axes[0, 0].set_ylabel("kHz")
    level_panel(axes[1, 0], ctx)

    ax = axes[1, 1]
    ax.plot(
        ctx["t_ramp"], ctx["speed_ramp"], lw=1.4, color=SPEED_COLOUR, label="smoothed mean speed"
    )
    ax.set_xlabel("s")
    ax.set_ylabel("mean rotor speed (rev/s)", color=SPEED_COLOUR)
    ax.tick_params(axis="y", labelcolor=SPEED_COLOUR)
    ax.grid(alpha=0.3)
    aw = ax.twinx()
    for i, (a, w) in enumerate(zip(ctx["model"].anchors, ctx["weight_ramp"], strict=True)):
        aw.plot(
            ctx["t_ramp"],
            w,
            lw=1.3,
            color=ANCHOR_COLOURS[i % len(ANCHOR_COLOURS)],
            label=f"w[{a.label}] ({a.speed_rps:.1f} rev/s)",
        )
    aw.plot(
        ctx["t_ramp"],
        1.0 - ctx["weight_ramp"].sum(axis=0),
        lw=1.1,
        ls="--",
        color="#7f7f7f",
        label="w[silence]",
    )
    aw.set_ylabel("blend weight")
    aw.set_ylim(-0.05, 1.08)
    aw.legend(fontsize=8, loc="center right")
    ax.set_title("speed and the anchor blend weights", fontsize=9)

    fig.suptitle(suptitle(ctx), fontsize=9)
    fig.tight_layout()
    path = OUT / f"flight_{ctx['rig']}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def weights_figure(models: dict[str, FM.FlightModel]) -> Path:
    """The blend weights as PURE FUNCTIONS of speed — what the assembly does."""
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.0), squeeze=False)
    for ax, (rig, model) in zip(axes[0], models.items(), strict=True):
        top = float(model.speeds[-1])
        s = np.linspace(0.0, 1.25 * top, 2001)
        w = FM.blend_weights(model, s)
        for i, (a, row) in enumerate(zip(model.anchors, w, strict=True)):
            ax.plot(
                s,
                row,
                lw=1.8,
                color=ANCHOR_COLOURS[i % len(ANCHOR_COLOURS)],
                label=f"{a.label} anchor ({a.speed_rps:.2f} rev/s)",
            )
        ax.plot(
            s, 1.0 - w.sum(axis=0), lw=1.4, ls="--", color="#7f7f7f", label="silence (implicit)"
        )
        for i, a in enumerate(model.anchors):
            ax.axvline(
                a.speed_rps, color=ANCHOR_COLOURS[i % len(ANCHOR_COLOURS)], lw=0.8, alpha=0.5
            )
            ax.annotate(
                f"{a.label}\n{a.speed_rps:.2f}",
                (a.speed_rps, 1.02),
                fontsize=8,
                ha="center",
                va="bottom",
                color=ANCHOR_COLOURS[i % len(ANCHOR_COLOURS)],
            )
        ax.set_xlabel("mean rotor speed s (rev/s)")
        ax.set_ylabel("weight")
        ax.set_ylim(-0.04, 1.18)
        ax.set_xlim(0.0, 1.25 * top)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="center right")
        ax.set_title(
            f"{rig}: hat weights over [0, {', '.join(f'{v:.1f}' for v in model.speeds)}]",
            fontsize=9,
        )
    fig.suptitle(
        "Blend weights against speed. Every anchor's weight is 0 at s = 0, so the blend is exactly "
        "silent there; above the top anchor the weights are held constant.",
        fontsize=9,
    )
    fig.tight_layout()
    path = OUT / "blend_weights.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def write_wavs(ctx: dict[str, Any]) -> dict[str, Any]:
    """Mic 0 of both arms under ONE playback gain; excerpt if the file is big."""
    stem = f"{ctx['kind']}_{ctx['rig']}"
    peak = max(float(np.abs(v[0]).max()) for v in ctx["waves"].values())
    playback = 0.7 / max(peak, 1e-12)
    written: list[str] = []
    for name, v in ctx["waves"].items():
        sf.write(OUT / f"{stem}_{name}.wav", (v[0] * playback).astype(np.float32), S2.SR)
        written.append(f"{stem}_{name}.wav")
    size_mb = max((OUT / w).stat().st_size for w in written) / 1e6
    excerpts: list[str] = []
    if size_mb > WAV_MAX_MB:
        # the last EXCERPT_S seconds: settled cruise on both rigs' windows
        cut = int(round(EXCERPT_S * S2.SR))
        for name, v in ctx["waves"].items():
            y = (v[0, -cut:] * playback).astype(np.float32)
            sf.write(OUT / f"{stem}_{name}_excerpt.wav", y, S2.SR)
            excerpts.append(f"{stem}_{name}_excerpt.wav")
    return dict(
        playback_gain=round(float(playback), 5),
        wavs=written,
        wav_max_mb=round(size_mb, 2),
        excerpts=excerpts,
        excerpt_s=EXCERPT_S if excerpts else None,
    )


def _segment_levels(levels: dict[str, np.ndarray], segs, out_key: str) -> dict[str, Any]:
    """Per-segment mean level of both arms, and the step between the ends."""
    out: dict[str, Any] = {out_key: {}}
    row: dict[str, Any]
    for name, a, b in segs:
        sl = slice(int(a / RAMP_S), int(b / RAMP_S))
        row = {arm: round(float(levels[arm][sl].mean()), 2) for arm in levels}
        row["model_minus_real_db"] = round(row["model"] - row["real"], 2)
        row["span_s"] = [a, b]
        out[out_key][name] = row
    if len(segs) >= 2:
        lo, hi = segs[0][0], segs[-1][0]
        real_step = out[out_key][hi]["real"] - out[out_key][lo]["real"]
        model_step = out[out_key][hi]["model"] - out[out_key][lo]["model"]
        out[f"{out_key}_step_db"] = dict(
            between=[lo, hi],
            real=round(real_step, 2),
            model=round(model_step, 2),
            model_minus_real=round(model_step - real_step, 2),
        )
    return out


def segment_stats(ctx: dict[str, Any]) -> dict[str, Any]:
    """Per-regime level agreement, plus the settled segment's LTAS shape.

    The whole-window RMS of a flight is not a useful number — it averages the
    regimes the assembly exists to distinguish. What matters is each regime's
    own level, and the STEP between them, because the step is the part the one
    global scalar cannot absorb.

    Measured twice: broadband, and above 300 Hz. FLY125's broadband level is
    dominated by sub-200 Hz rumble that is present with the rotors stopped and
    that the model does not generate at all, so the broadband step between
    standby and cruise is compressed by content that is not rotor noise. The
    high-passed step is the one to read for a comb-and-floor model; both are
    reported because neither is the whole story.
    """
    from scipy.signal import butter, sosfiltfilt

    from experiments.stochastic_fit import accept_stats as st

    segs = FLIGHT_SEGMENTS[ctx["rig"]]
    out: dict[str, Any] = _segment_levels(ctx["levels"], segs, "segments")
    sos = butter(4, 300.0, "hp", fs=S2.SR, output="sos")
    hp = {arm: ramp_db(sosfiltfilt(sos, v, axis=-1)) for arm, v in ctx["waves"].items()}
    out.update(_segment_levels(hp, segs, "segments_hp300"))
    out["highpass_hz"] = 300.0
    # the legacy key the earlier index files used, kept so a reader of both
    # generations compares the same quantity
    if "segments_step_db" in out:
        out["level_step_db"] = out["segments_step_db"]
    name, a, b = segs[-1]
    sl = slice(int(a * S2.SR), int(b * S2.SR))
    bands = {arm: st.ltas_bands(ctx["waves"][arm][0, sl]) for arm in ctx["waves"]}
    dev = np.asarray(bands["model"]) - np.asarray(bands["real"])
    out["settled_ltas"] = dict(
        segment=name,
        span_s=[a, b],
        bands_hz=[list(x) for x in st.BANDS],
        model_minus_real_db=np.round(dev, 2).tolist(),
        mean_abs_db=round(float(np.abs(dev).mean()), 2),
    )
    # ltas_bands is SHAPE-only (dB re 200-400 Hz) and starts at 100 Hz, so it
    # cannot see an absolute low-frequency excess at all. These are ABSOLUTE
    # band levels on a grid that reaches 20 Hz, which is what localises a
    # broadband level disagreement the shape statistic is blind to.
    edges = ABSOLUTE_BAND_EDGES
    spec: dict[str, list[float]] = {}
    power: dict[str, np.ndarray] = {}
    freqs = np.zeros(0)
    for arm in ctx["waves"]:
        x = ctx["waves"][arm][0, sl]
        nfft = 4096
        win = np.hanning(nfft + 1)[:nfft]
        fr = np.stack([x[i : i + nfft] * win for i in range(0, x.size - nfft, nfft // 2)])
        p = (np.abs(np.fft.rfft(fr, axis=-1)) ** 2).mean(axis=0)
        f = np.fft.rfftfreq(nfft, 1.0 / S2.SR)
        power[arm], freqs = p, f
        spec[arm] = [
            10.0 * np.log10(float(p[(f >= lo) & (f < hi)].sum()) + 1e-30)
            for lo, hi in zip(edges, edges[1:], strict=False)
        ]
    out["settled_absolute_bands"] = dict(
        segment=name,
        span_s=[a, b],
        edges_hz=list(edges),
        real_db=np.round(spec["real"], 2).tolist(),
        model_db=np.round(spec["model"], 2).tolist(),
        model_minus_real_db=np.round(
            np.asarray(spec["model"]) - np.asarray(spec["real"]), 2
        ).tolist(),
    )
    # LINES or FLOOR? The comb bins carry the lines; the HALF-integer orders are
    # the null where no rotor line can exist (accept_stats' own convention), so
    # the two together separate a line-level error from a floor-level error.
    speed = float(ctx["speed_ramp"][int(a / RAMP_S) : int(b / RAMP_S)].mean())
    df = float(freqs[1] - freqs[0])
    probe: dict[str, dict[str, list[float]]] = {}
    for arm, p in power.items():
        got: dict[str, list[float]] = {"line_db": [], "null_db": []}
        for k in LOW_ORDER_PROBE:
            for key, order in (("line_db", float(k)), ("null_db", k + 0.5)):
                c = int(round(order * speed / df))
                seg = p[max(c - 2, 0) : c + 3]
                got[key].append(round(float(10.0 * np.log10(seg.max() + 1e-30)), 2))
        probe[arm] = got
    out["settled_low_orders"] = dict(
        segment=name,
        mean_speed_rps=round(speed, 2),
        orders=list(LOW_ORDER_PROBE),
        line_hz=[round(k * speed, 1) for k in LOW_ORDER_PROBE],
        real=probe["real"],
        model=probe["model"],
        line_model_minus_real_db=[
            round(m - r, 2)
            for m, r in zip(probe["model"]["line_db"], probe["real"]["line_db"], strict=True)
        ],
        null_model_minus_real_db=[
            round(m - r, 2)
            for m, r in zip(probe["model"]["null_db"], probe["real"]["null_db"], strict=True)
        ],
    )
    return out


def comb_track(ctx: dict[str, Any]) -> dict[str, Any]:
    """Strongest line in a band, per step, for both arms, with implied order.

    A single-peak tracker is honest only about where it agrees: it HOPS between
    orders whenever two lines trade dominance, so the transient rows are not a
    sweep measurement. The settled rows are, and that is what is read.
    """
    (t0, t1, step, f_lo, f_hi), excursion = COMB_TRACK[ctx["rig"]]
    n = 2048
    w = np.hanning(n + 1)[:n]
    f = np.fft.rfftfreq(n, 1.0 / S2.SR)
    lo, hi = int(np.searchsorted(f, f_lo)), int(np.searchsorted(f, f_hi))
    # a frame must fit inside the clip: the last requested step can sit within
    # one window of the end, and a short slice would not broadcast
    limit = int(ctx["waves"]["real"].shape[-1])
    times = np.arange(t0, t1, step)
    times = times[(times * S2.SR).astype(np.int64) + n <= limit]
    speed = np.array([ctx["render"].speed_rps[int(t * S2.SR)] for t in times])
    peaks: dict[str, list[float]] = {}
    for arm, x in ctx["waves"].items():
        got = []
        for t in times:
            i = int(t * S2.SR)
            spec = np.abs(np.fft.rfft(x[0, i : i + n] * w))[lo:hi]
            got.append(float(f[lo + int(np.argmax(spec))]))
        peaks[arm] = [round(v, 1) for v in got]
    order = {arm: np.asarray(peaks[arm]) / np.maximum(speed, 1e-9) for arm in peaks}
    settled = times >= times[-1] - 0.4
    return dict(
        t_s=np.round(times, 2).tolist(),
        band_hz=[f_lo, f_hi],
        speed_rps=np.round(speed, 2).tolist(),
        peak_hz=peaks,
        implied_order={arm: np.round(v, 1).tolist() for arm, v in order.items()},
        settled_window_s=[round(float(times[-1] - 0.4), 2), round(float(times[-1]), 2)],
        settled_implied_order={arm: round(float(v[settled].mean()), 2) for arm, v in order.items()},
        speed_excursion=None
        if excursion is None
        else dict(
            peak_window_s=list(excursion[0]),
            trough_window_s=list(excursion[1]),
            peak_rps=round(
                float(
                    ctx["render"]
                    .speed_rps[int(excursion[0][0] * S2.SR) : int(excursion[0][1] * S2.SR)]
                    .max()
                ),
                2,
            ),
            trough_rps=round(
                float(
                    ctx["render"]
                    .speed_rps[int(excursion[1][0] * S2.SR) : int(excursion[1][1] * S2.SR)]
                    .min()
                ),
                2,
            ),
        ),
    )


def _render_physical(
    export: dict[str, Any],
    rps: np.ndarray,
    *,
    n_mics: int,
    seed: int = SEED,
    line_mode: str = "fm",
    sample_rate_work: int = 44100,
) -> np.ndarray:
    """``stage2.render_from_export``'s body on an ALREADY-physical export.

    Needed because :func:`flight_model.physical_export` is not idempotent —
    calling it twice would apply the render-units term twice — and the
    diagnosis has to render hand-modified physical exports.
    """
    from data_processing import stochastic_rotor_noise as srn

    rps = np.atleast_2d(np.asarray(rps, dtype=np.float64))
    params = S2.params_from_export(
        export, rps.mean(axis=1), sample_rate=sample_rate_work, n_mics=n_mics
    )
    n_work = int(round(rps.shape[-1] / S2.SR * sample_rate_work))
    t_src = np.linspace(0.0, 1.0, rps.shape[-1])
    t_dst = np.linspace(0.0, 1.0, n_work)
    rps_work = np.stack([np.interp(t_dst, t_src, r) for r in rps])
    audio, _ = srn.synthesize(
        params,
        rps_work,
        rng=np.random.default_rng(seed),
        n_mics=n_mics,
        line_mode=line_mode,
        n_fft=1 << 16,
        normalize_rms=None,
    )
    audio = S2.antialias(np.asarray(audio, dtype=np.float64), sample_rate_work)
    clip = C.Clip(
        "synthetic", "synthetic", np.asarray(audio, np.float32), rps_work, sample_rate_work
    )
    return np.asarray(C.decimate(clip, S2.SR).audio, dtype=np.float64)


def _coherent_share(export: dict[str, Any], n_orders: int) -> np.ndarray:
    """``synthesize``'s own coherent share ``w_k = exp(-(k/k_half)^2)``."""
    k = np.arange(1, n_orders + 1, dtype=np.float64)
    k_half = float(export.get("coherence_k_half", 0.0))
    if k_half <= 0.0:
        return np.ones(n_orders, dtype=np.float64)
    return np.clip(np.exp(-((k / k_half) ** 2)), 1e-6, 1.0 - 1e-6)


def _split_outside(
    export: dict[str, Any], rps: np.ndarray, *, n_mics: int, seed: int = SEED
) -> np.ndarray:
    """The coherent/incoherent split done OUTSIDE ``synthesize``.

    Two independent renders: the coherent share as a tone bank
    (``coherence_k_half = 0``, ``line_mode="fm"``) and the incoherent share as
    filtered noise (``line_mode="stochastic"``), each carrying HALF the floor
    power so the two sum to the right total. Neither arm performs a split, so
    inside each one ``want`` and ``have`` refer to the same spectrum, which is
    exactly the property the shipped code loses. This is a MEASUREMENT of what
    the corrected mixing would produce, not a patch.
    """
    prof = np.atleast_2d(np.asarray(export["profile_db"], dtype=np.float64))
    w = _coherent_share(export, prof.shape[1])
    half_floor = float(export["floor_mean_db"]) - 10.0 * np.log10(2.0)
    coh = dict(
        export,
        coherence_k_half=0.0,
        floor_mean_db=half_floor,
        profile_db=(prof + 10.0 * np.log10(w)[None, :]).tolist(),
    )
    inc = dict(
        export,
        coherence_k_half=0.0,
        floor_mean_db=half_floor,
        profile_db=(prof + 10.0 * np.log10(1.0 - w)[None, :]).tolist(),
    )
    a = _render_physical(coh, rps, n_mics=n_mics, seed=seed, line_mode="fm")
    b = _render_physical(inc, rps, n_mics=n_mics, seed=seed + 1, line_mode="stochastic")
    n = min(a.shape[-1], b.shape[-1])
    return a[:, :n] + b[:, :n]


def _coherent_bank_inflation_db(
    export: dict[str, Any], rps: np.ndarray, *, n_mics: int, sample_rate_work: int = 44100
) -> np.ndarray:
    """``(M,)`` dB the shipped mixing adds to the coherent tone bank.

    ``synthesize`` adds the INCOHERENT share of every order into ``floor_spec``
    (``stochastic_rotor_noise.py:1670``) and realizes ``floor_audio`` from it,
    but builds ``want``'s denominator from ``psd["floor"]`` alone (``:1685``,
    ``:1691-1693``) while ``have``'s denominator is ``var(floor_audio)``
    (``:1694``). The two sides of the ratio therefore refer to different
    spectra, and ``scale = sqrt(want/have)`` (``:1695``) comes out high by
    exactly the factor the incoherent comb raises the broadband floor by:

        I = 1 + mean(gains[m] . incoherent) / (mean(psd["floor"]) * floor_mic[m])

    That is a single factor on the whole tone bank, so the per-ORDER error is
    ``10 log10(w_k I + (1 - w_k))`` — large only where the coherent share
    ``w_k`` is large, i.e. at the first two or three orders.
    """
    from data_processing import stochastic_rotor_noise as srn

    rps = np.atleast_2d(np.asarray(rps, dtype=np.float64))
    p = S2.params_from_export(export, rps.mean(axis=1), sample_rate=sample_rate_work, n_mics=n_mics)
    n_fft = 1 << 16
    hop, pad = n_fft // 4, n_fft
    n_work = int(round(rps.shape[-1] / S2.SR * sample_rate_work))
    t_src = np.linspace(0.0, 1.0, rps.shape[-1])
    t_dst = np.linspace(0.0, 1.0, n_work)
    rps_work = np.stack([np.interp(t_dst, t_src, r) for r in rps])
    n_frames = 1 + int(np.ceil(max(n_work + 2 * pad - n_fft, 0) / hop))
    frame_t = (np.arange(n_frames) * hop + n_fft / 2.0 - pad) / sample_rate_work
    clip_t = np.arange(n_work) / sample_rate_work
    rf = np.stack(
        [
            np.interp(frame_t, clip_t, rps_work[r], left=rps_work[r][0], right=rps_work[r][-1])
            for r in range(rps_work.shape[0])
        ]
    )
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate_work)
    psd = srn.build_psd(p, rf, freqs, dt=hop / sample_rate_work, rng=np.random.default_rng(0))
    w = _coherent_share(export, p.n_harmonics)
    prof = np.atleast_2d(np.asarray(p.profile_db, dtype=np.float64))
    inc = srn.build_psd(
        p.with_(profile_db=prof + 10.0 * np.log10(1.0 - w)[None, :]),
        rf,
        freqs,
        dt=hop / sample_rate_work,
        rng=np.random.default_rng(0),
    )["lines"]
    gains = 10.0 ** (np.asarray(p.fixed_mic_gain_db, dtype=np.float64) / 10.0)
    floor_mic = 10.0 ** (np.asarray(p.fixed_mic_floor_db, dtype=np.float64) / 10.0)
    floor_mean = float(np.mean(psd["floor"]))
    return np.array(
        [
            10.0
            * np.log10(
                1.0
                + float(np.mean(np.tensordot(gains[m], inc, axes=(0, 0))))
                / (floor_mean * float(floor_mic[m]))
            )
            for m in range(n_mics)
        ]
    )


def low_order_diagnosis(rig: str, *, mic: int = 0) -> dict[str, Any]:
    """THREE CURVES per order: the real clip, the fit's own forward law, the render.

    Run on the anchor's OWN fit support, so nothing here is attributable to the
    blend. The three quantities are the mean periodogram of

    * the REAL clip on the fit's own front end (16384-point Hann, no overlap);
    * ``revised_eval.predicted_m`` — the legacy family's own forward law for
      the same physical export, i.e. what the FIT says the periodogram is;
    * ``stage2.render_from_export``'s output, re-analysed on the same grid.

    If (2) tracks (1) and (3) does not, the error is in the RENDER path; if (2)
    and (3) agree and both sit above (1), it is in the FIT. A fourth curve is
    added because the answer turned out to be the render: the same render with
    the coherent/incoherent split performed consistently
    (:func:`_split_outside`), which is what the one-line correction would give.

    ``predicted_m`` gets the ``_to_physical`` export and NOT the render-units
    term: it predicts the ANALYSIS-grid periodogram directly, while the term
    belongs to the work-grid render. It is asked to apply the render transfer,
    because the render physically does.
    """
    label, recording, start_s, seconds = DIAGNOSIS_SUPPORT[rig]
    path, selector = next((p, s) for p, s, lab, *_ in ANCHORS[rig] if lab == label)
    entry = prev_entry(path, selector)
    clip = load_real(rig, recording, start_s, seconds, f"diag_{rig}")
    pg = C_DATA.periodogram(clip, n_fft=DIAGNOSIS_N_FFT, hop=DIAGNOSIS_N_FFT)
    rps = np.asarray(clip.rps, dtype=np.float64)
    n_mics = int(clip.audio.shape[0])
    export = FM.physical_export(entry)

    def mean_mic(power: np.ndarray) -> np.ndarray:
        return np.asarray(power, dtype=np.float64)[mic].mean(axis=0)

    from experiments.stochastic_fit import revised_eval as RE

    mp = RE.ModelParams(
        params=RE._to_physical(entry),
        spec=json.loads(Path(path).read_text())["spec"],
        source={},
    )
    curves = {
        "real": mean_mic(pg.power),
        "predicted_m": mean_mic(RE.predicted_m(mp, pg, n_mics=n_mics, apply_transfer=True)),
    }
    for name, audio in (
        ("render", _render_physical(export, rps, n_mics=n_mics)),
        ("render_split_outside", _split_outside(export, rps, n_mics=n_mics)),
    ):
        rp = C_DATA.periodogram(
            C.Clip("s", "s", audio.astype(np.float32), rps, S2.SR),
            n_fft=DIAGNOSIS_N_FFT,
            hop=DIAGNOSIS_N_FFT,
        )
        curves[name] = mean_mic(rp.power)

    f = np.asarray(pg.freqs, dtype=np.float64)
    rate = float(rps.mean())
    infl = _coherent_bank_inflation_db(export, rps, n_mics=n_mics)
    w = _coherent_share(export, int(np.asarray(export["profile_db"]).shape[1]))
    i_lin = 10.0 ** (float(infl[mic]) / 10.0)
    rows = []
    for k in DIAGNOSIS_ORDERS:
        m = (f >= (k - 0.35) * rate) & (f < (k + 0.35) * rate)
        ref = float(curves["predicted_m"][m].sum())
        rows.append(
            dict(
                band=f"k={k}",
                centre_hz=round(k * rate, 1),
                coherent_share=round(float(w[k - 1]), 4),
                **{
                    n: round(float(10.0 * np.log10(float(v[m].sum()) / ref)), 2)
                    for n, v in curves.items()
                },
                predicted_render_error_db=round(
                    float(10.0 * np.log10(w[k - 1] * i_lin + (1.0 - w[k - 1]))), 2
                ),
            )
        )
    for lo, hi in DIAGNOSIS_BANDS:
        m = (f >= lo) & (f < hi)
        ref = float(curves["predicted_m"][m].sum())
        rows.append(
            dict(
                band=f"{lo}-{hi} Hz",
                centre_hz=None,
                coherent_share=None,
                **{
                    n: round(float(10.0 * np.log10(float(v[m].sum()) / ref)), 2)
                    for n, v in curves.items()
                },
                predicted_render_error_db=None,
            )
        )
    return dict(
        rig=rig,
        anchor=label,
        fit=str(path),
        selector=selector,
        support=f"{recording}@{start_s:.3f}+{seconds:g}",
        mic=mic,
        n_fft=DIAGNOSIS_N_FFT,
        mean_rate_rps=round(rate, 3),
        reference="predicted_m (the legacy fit's own forward law), 0 dB",
        coherence_k_half=round(float(export.get("coherence_k_half", 0.0)), 4),
        coherent_bank_inflation_db=np.round(infl, 2).tolist(),
        coherent_bank_inflation_db_this_mic=round(float(infl[mic]), 2),
        gamma0=round(float(np.mean(entry["params"]["gamma0"])), 4),
        gamma_slope=round(float(np.mean(entry["params"]["gamma_slope"])), 4),
        rows=rows,
    )


def floor_divergence(rig: str, n_mics: int = 8) -> dict[str, Any] | None:
    """Evidence that a rebased anchor really was unrenderable, or ``None``.

    For every anchor whose floor exponent :func:`flight_model.assemble` had to
    rebase, render the export WITHOUT that rebase — physical levels and all,
    so the only difference from the model's own anchor is the exponent — over
    the rig's startup window, and count how much of it is not finite. This is
    the claim the rebase rests on, so it is measured rather than asserted: a
    negative exponent is only a problem if the trajectory actually reaches
    zero speed, and here it does. The count is scale-invariant anyway —
    ``inf`` stays ``inf`` under any gain.
    """
    model, provenance = build_model(rig)
    rebased = [p for p in provenance if p["floor_exp_rebased"]]
    if not rebased:
        return None
    recording, start_s, seconds, _ = WINDOWS["startup"][rig]
    real = load_real(rig, recording, start_s, seconds, f"divergence_{rig}")
    rps = np.asarray(real.rps, dtype=np.float64)
    rows = {}
    for p in rebased:
        unrebased = FM.physical_export(prev_entry(Path(p["fit"]), p["selector"]))
        with np.errstate(divide="ignore", invalid="ignore"):
            bad = S2.render_from_export(
                unrebased, rps, n_mics=n_mics, seed=SEED, normalize_rms=None
            )
        n_bad = int((~np.isfinite(bad)).sum())
        rows[p["label"]] = dict(
            floor_exp_fitted=p["floor_exp_fitted"],
            window=f"{recording}@{start_s:.3f}+{seconds:g}",
            zero_telemetry_fraction=round(float((rps == 0.0).mean()), 5),
            nonfinite_samples=n_bad,
            total_samples=int(bad.size),
            nonfinite_fraction=round(n_bad / bad.size, 5),
        )
    return rows


def verify(rig: str, n_mics: int = 8) -> dict[str, Any]:
    """The mechanism checks the assembly's honesty rests on.

    * PHASE LOCK — render every anchor on ONE constant-speed trajectory with
      the shared seed and correlate them above 500 Hz, then correlate one
      anchor against itself under a DIFFERENT seed. High same-seed correlation
      with near-zero different-seed correlation is what makes the time-domain
      blend an amplitude interpolation rather than a sum of two noises. It is a
      two-anchor statement, so a single-anchor rig has nothing to report.
    * BLEND AT AN ANCHOR — evaluate the blend exactly at one anchor's speed and
      difference it against that anchor's own render. It must come back to
      floating-point noise, which is what proves the weights are a partition
      and not a rescaling.
    * FLOOR DIVERGENCE — :func:`floor_divergence`, the measured justification
      for any floor-exponent rebase this rig needed.
    """
    from scipy.signal import butter, sosfiltfilt

    model, _ = build_model(rig)
    speed = float(model.speeds[0])
    rps = np.full((model.n_rotors, 2 * S2.SR), speed)
    renders = [
        S2.render_from_export(a.export, rps, n_mics=n_mics, seed=SEED, normalize_rms=None)
        for a in model.anchors
    ]
    sos = butter(4, 500.0, "hp", fs=S2.SR, output="sos")

    def corr(x: np.ndarray, y: np.ndarray) -> list[float]:
        a, b = sosfiltfilt(sos, x), sosfiltfilt(sos, y)
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        num = (a * b).sum(axis=1)
        return np.round(num / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1)), 3).tolist()

    out: dict[str, Any] = dict(
        rig=rig,
        speed_rps=round(speed, 4),
        seed=SEED,
        other_seed=SEED + 1,
        highpass_hz=500.0,
        n_orders=model.n_orders,
    )
    other = S2.render_from_export(
        model.anchors[-1].export, rps, n_mics=n_mics, seed=SEED + 1, normalize_rms=None
    )
    out["different_seed_per_mic_corr"] = corr(renders[-1], other)
    if len(renders) >= 2:
        out["same_seed_per_mic_corr"] = corr(renders[0], renders[-1])
        out["same_seed_pair"] = [model.anchors[0].label, model.anchors[-1].label]
    else:
        out["same_seed_per_mic_corr"] = None
        out["same_seed_pair"] = None
    blend = FM.render_flight(model, rps, n_mics=n_mics, seed=SEED)
    n = min(blend.audio.shape[-1], renders[0].shape[-1])
    out["blend_at_anchor"] = dict(
        anchor=model.anchors[0].label,
        weights=np.format_float_scientific(blend.weights[0, 0], precision=8),
        other_weights=[np.format_float_scientific(w, precision=3) for w in blend.weights[1:, 0]],
        # the 50 ms moving average of a CONSTANT trajectory is not bit-exactly
        # constant (cumulative-sum rounding), so the hat weight lands within a
        # few 1e-14 of 1 rather than on it — which is the whole residual below
        weight_max_dev_from_one=float(np.max(np.abs(blend.weights[0] - 1.0))),
        max_abs_residual=float(np.max(np.abs(blend.audio[:, :n] - renders[0][:, :n]))),
        anchor_rms=round(float(np.sqrt(np.mean(np.square(renders[0][:, :n])))), 8),
        anchor_peak=round(float(np.max(np.abs(renders[0][:, :n]))), 6),
    )
    out["floor_divergence"] = floor_divergence(rig, n_mics)
    return out


def index_row(ctx: dict[str, Any], sc: dict[str, Any], audio: dict[str, Any], figure: str) -> dict:
    waves, levels = ctx["waves"], ctx["levels"]
    return dict(
        rig=ctx["rig"],
        window=ctx["kind"],
        recording=ctx["recording"],
        start_s=ctx["start_s"],
        seconds=ctx["seconds"],
        dataset=DATASET[ctx["rig"]],
        rps_key=RAW_KEY[ctx["rig"]],
        anchors=ctx["provenance"],
        seed=SEED,
        n_orders=ctx["model"].n_orders,
        n_mics=ctx["n_mics"],
        physical_level_residual_db=round(float(-20 * np.log10(ctx["gain"])), 3),
        match_gain_db_would_be=round(float(20 * np.log10(ctx["gain"])), 3),
        match_gain_applied=False,
        match_window_s=ctx["match_s"],
        rms_dbfs={k: round(dbfs(v), 2) for k, v in waves.items()},
        ramp_s=RAMP_S,
        ramp_db={k: np.round(v, 2).tolist() for k, v in levels.items()},
        speed_rps_ramp=np.round(ctx["speed_ramp"], 2).tolist(),
        weight_ramp={
            a.label: np.round(w, 4).tolist()
            for a, w in zip(ctx["model"].anchors, ctx["weight_ramp"], strict=True)
        },
        onset_s=ctx["onsets"],
        onset_base_db=ctx["onset_base"],
        onset_threshold_db=ONSET_DB,
        ramp_correlation=round(ctx["corr"], 4),
        silence_check=sc,
        diagnostics=ctx["render"].diagnostics,
        figure=figure,
        **audio,
        **(segment_stats(ctx) if ctx["kind"] == "flight" else {"comb_track": comb_track(ctx)}),
    )


def report(ctx: dict[str, Any], sc: dict[str, Any], row: dict[str, Any]) -> None:
    ctx_id = f"{ctx['recording']}@{ctx['start_s']:.3f}+{ctx['seconds']:g}"
    print(f"=== {ctx['rig']} {ctx['kind']}: {ctx_id}", flush=True)
    print(
        "  anchors      "
        + ", ".join(f"{p['label']} {p['speed_rps']:.2f} rev/s" for p in ctx["provenance"])
        + f"   K={ctx['model'].n_orders}  seed={SEED}"
    )
    for p in ctx["provenance"]:
        print(
            f"  exponents    {p['label']}: amp_exp {p['amp_exp_fitted']:+.4f}  "
            f"floor_exp fitted {p['floor_exp_fitted']:+.4f} -> used {p['floor_exp_used']:+.4f}  "
            + (
                "REBASED (fitted floor law diverges at zero speed)"
                if p["floor_exp_rebased"]
                else "as fitted, no rebase"
            )
        )
    for label, notes in ctx["render"].diagnostics["anchor_notes"].items():
        for note in notes:
            print(f"  NOTE {label}: {note}")
    for p in ctx["provenance"]:
        print(
            f"  level path   {p['label']}: power_scale {p['power_scale']:.6e} "
            f"({p['power_scale_folded_db']:+.3f} dB) folded + render units "
            f"{p['render_units_rate_factor_db']:+.3f} dB at "
            f"{p['render_units_sample_rate_work']} Hz work grid"
        )
    print(
        "  regions      "
        + ", ".join(f"{k} {v:.3f}" for k, v in ctx["render"].diagnostics["region_fraction"].items())
    )
    print(
        f"  silence      {'PASS' if sc['exact'] else 'FAIL'} "
        f"(peak |x| = {sc['peak_abs']!r}, max weight {sc['max_weight']!r}, "
        f"max smoothed speed {sc['max_speed_rps']!r})"
    )
    if not sc["exact"]:
        print(
            "               NOT exactly zero: a zero blend weight cannot silence a "
            "non-finite or non-zero-weighted anchor render"
        )
    print(
        f"  rms dBFS     real {row['rms_dbfs']['real']:.2f}  model {row['rms_dbfs']['model']:.2f}"
        f"   (physical residual over last {ctx['match_s']:g} s: "
        f"{row['physical_level_residual_db']:+.2f} dB, NO gain applied)"
    )
    onsets, base = ctx["onsets"], ctx["onset_base"]
    print(
        "  onset        real "
        + ("n/a" if onsets["real"] is None else f"{onsets['real']:.2f} s")
        + "  model "
        + ("n/a" if onsets["model"] is None else f"{onsets['model']:.2f} s")
        + f"   (+{ONSET_DB:g} dB over own first 100 ms: real base {base['real']:.2f} dB, "
        f"model base {base['model']:.2f} dB)"
    )
    print(f"  ramp corr    {ctx['corr']:.4f}")
    if "segments" in row:
        for name, seg in row["segments"].items():
            print(
                f"  {name:<12} real {seg['real']:.2f}  model {seg['model']:.2f}  "
                f"model-real {seg['model_minus_real_db']:+.2f} dB "
                f"over [{seg['span_s'][0]:g}, {seg['span_s'][1]:g}) s"
            )
        if "level_step_db" in row:
            st = row["level_step_db"]
            print(
                f"  level step   {st['between'][0]} -> {st['between'][1]}: real "
                f"{st['real']:+.2f} dB, model {st['model']:+.2f} dB, "
                f"model-real {st['model_minus_real']:+.2f} dB"
            )
        ab = row["settled_absolute_bands"]
        print(
            "  abs bands    "
            + ", ".join(
                f"{lo:g}-{hi:g}Hz {d:+.1f}"
                for lo, hi, d in zip(
                    ab["edges_hz"], ab["edges_hz"][1:], ab["model_minus_real_db"], strict=False
                )
            )
            + "  (model-real, absolute)"
        )
        lo_ = row["settled_low_orders"]
        print(
            f"  lines k @{lo_['mean_speed_rps']:.1f} rev/s  "
            + ", ".join(
                f"k{k} ({hz:.0f}Hz) {d:+.1f}"
                for k, hz, d in zip(
                    lo_["orders"], lo_["line_hz"], lo_["line_model_minus_real_db"], strict=True
                )
            )
        )
        print(
            "  nulls k+0.5  "
            + ", ".join(
                f"k{k}.5 {d:+.1f}"
                for k, d in zip(lo_["orders"], lo_["null_model_minus_real_db"], strict=True)
            )
            + "  (model-real)"
        )
        for name, seg in row["segments_hp300"].items():
            print(
                f"  {name + ' >300Hz':<14} real {seg['real']:.2f}  model {seg['model']:.2f}  "
                f"model-real {seg['model_minus_real_db']:+.2f} dB"
            )
        if "segments_hp300_step_db" in row:
            st = row["segments_hp300_step_db"]
            print(
                f"  step >300Hz  {st['between'][0]} -> {st['between'][1]}: real "
                f"{st['real']:+.2f} dB, model {st['model']:+.2f} dB, "
                f"model-real {st['model_minus_real']:+.2f} dB"
            )
        lt = row["settled_ltas"]
        print(
            f"  settled LTAS {lt['segment']}: mean|dev| {lt['mean_abs_db']:.2f} dB, "
            f"per band {lt['model_minus_real_db']}"
        )
    if "comb_track" in row:
        ct = row["comb_track"]
        print(
            f"  comb settled implied order over {ct['settled_window_s']} s "
            f"({ct['band_hz'][0]:.0f}-{ct['band_hz'][1]:.0f} Hz): "
            + ", ".join(f"{k} {v:.2f}" for k, v in ct["settled_implied_order"].items())
        )
        ex = ct["speed_excursion"]
        if ex is not None:
            print(
                f"  speed excursion peak {ex['peak_rps']:.2f} rev/s over "
                f"{ex['peak_window_s']} s -> trough {ex['trough_rps']:.2f} rev/s over "
                f"{ex['trough_window_s']} s ({ex['peak_rps'] / ex['trough_rps']:.2f}x)"
            )
    print(f"  audio        {', '.join(row['wavs'])}  (max {row['wav_max_mb']:.2f} MB)")
    if row["excerpts"]:
        print(f"  excerpts     {', '.join(row['excerpts'])}  (last {EXCERPT_S:g} s)")
    print(f"  wrote        {row['figure']}, {ctx['kind']}_{ctx['rig']}.json")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = {"startup": startup_figure, "flight": flight_figure}
    checks: dict[str, Any] = {}
    for rig in RIGS:
        model, _ = build_model(rig)
        sc = silence_check(model, 8)
        for kind in ("startup", "flight"):
            ctx = compare(rig, kind)
            fig_path = figures[kind](ctx)
            audio = write_wavs(ctx)
            row = index_row(ctx, sc, audio, fig_path.name)
            (OUT / f"{kind}_{rig}.json").write_text(json.dumps(row, indent=1))
            report(ctx, sc, row)
        v = verify(rig)
        v["silence_check"] = sc
        checks[rig] = v
        print(f"=== {rig} mechanism checks")
        print(
            f"  same-seed anchor corr per mic  {v['same_seed_pair']}: {v['same_seed_per_mic_corr']}"
        )
        print(f"  diff-seed anchor corr per mic  {v['different_seed_per_mic_corr']}")
        b = v["blend_at_anchor"]
        print(
            f"  blend at {b['anchor']} anchor: weight {b['weights']} "
            f"(others {b['other_weights']}), max|blend - anchor| {b['max_abs_residual']:.3e}, "
            f"anchor RMS {b['anchor_rms']:.6f}"
        )
        for label, fd in (v["floor_divergence"] or {}).items():
            print(
                f"  floor divergence {label}: floor_exp {fd['floor_exp_fitted']:+.3f} on "
                f"{fd['window']} — {fd['zero_telemetry_fraction'] * 100:.1f}% exactly-zero "
                f"telemetry samples, unmodified render has {fd['nonfinite_samples']}/"
                f"{fd['total_samples']} non-finite samples "
                f"({fd['nonfinite_fraction'] * 100:.1f}%)"
            )
    (OUT / "verify.json").write_text(json.dumps(checks, indent=1))

    diagnosis = {}
    for rig in RIGS:
        d = low_order_diagnosis(rig)
        diagnosis[rig] = d
        print(f"=== {rig} low-order diagnosis on {d['support']} (mic {d['mic']})")
        print(
            f"  coherence_k_half {d['coherence_k_half']:.4f}, gamma0 {d['gamma0']:.3f}, "
            f"gamma_slope {d['gamma_slope']:.3f}, rate {d['mean_rate_rps']:.2f} rev/s, "
            f"coherent-bank inflation {d['coherent_bank_inflation_db_this_mic']:+.2f} dB"
        )
        print(
            f"  {'band':>12} {'w_k':>6} {'real':>8} {'render':>8} {'fixed':>8} {'theory':>8}"
            "   (dB re predicted_m)"
        )
        for r in d["rows"]:
            wk = "" if r["coherent_share"] is None else f"{r['coherent_share']:.3f}"
            th = (
                ""
                if r["predicted_render_error_db"] is None
                else f"{r['predicted_render_error_db']:+.2f}"
            )
            print(
                f"  {r['band']:>12} {wk:>6} {r['real']:8.2f} {r['render']:8.2f} "
                f"{r['render_split_outside']:8.2f} {th:>8}"
            )
    (OUT / "low_order_diagnosis.json").write_text(json.dumps(diagnosis, indent=1))
    path = weights_figure({rig: build_model(rig)[0] for rig in RIGS})
    print(f"wrote {path}, {OUT / 'verify.json'} and {OUT / 'low_order_diagnosis.json'}")


if __name__ == "__main__":
    main()
