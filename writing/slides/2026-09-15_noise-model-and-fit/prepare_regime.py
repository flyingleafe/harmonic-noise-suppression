"""Real-data prediction figures for the regime slide of this deck.

One PNG per chosen validation clip:

  * top    — the clip's mic-0 spectrogram (kHz against seconds);
  * middle — a strip of the four flight regimes over the same time axis, so the
             viewer can see which part of the clip is zero / standby / ramp /
             cruise before reading the tracks;
  * bottom — the four TARGET rotor-speed tracks as thick light lines with the
             predicted tracks of all three models drawn thin on top.

DRAWN RAW, SCORED WITH PIT. Each model emits four rotor-speed series and those
four series are drawn exactly as emitted: no permutation, no assignment. A
model's line may therefore sit near a different target track than "its own"
rotor, which is the truth about what the model outputs. The reported MAE is
the project's per-frame PIT metric (:func:`_regime_decomp.pit_align`,
unchanged), which is permutation-invariant and so does not care how the rows
are ordered. Nothing in this file solves an assignment.

The regime labelling and the checkpoint selection are imported from
``scripts/_regime_decomp.py`` — the figures and the table are the same numbers,
computed once.

    PYTHONPATH="$PWD/src:$PWD/scripts" python \
        writing/slides/2026-09-15_noise-model-and-fit/prepare_regime.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = HERE / "assets"
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _regime_decomp import (  # noqa: E402
    FPS,
    REGIMES,
    VALID,
    frame_regimes4,
    pit_abs_error,
    predict_clip,
)

DECOMP_JSON = ROOT / "results/regime_decomp/scv2_regimes.json"

#: Every model drawn on every figure, with the colour and label each keeps
#: everywhere. ``real-trained`` is ``real_r4_scv2_unified``, the reference the
#: whole comparison is against: it trained on real data, so the distance from a
#: synthetic arm's line to ITS line — not to the target — is what the transfer
#: question is about. Near-black keeps it separable from both arm colours and
#: from the light target band.
ARMS = (
    ("rig_easy_scv2_unified", "easy", "#1f77b4"),
    ("rig_hard_scv2_unified", "hard", "#d62728"),
    ("real_r4_scv2_unified", "real-trained", "#111111"),
)
CKPT = "best_real_overall"

#: Regime strip colours — light enough to sit under text, distinct in grayscale
#: order (zero pale, cruise strongest).
REGIME_COLOUR = {
    "zero": "#e8eef5",
    "standby": "#bcd4ea",
    "ramp": "#f7c98b",
    "cruise": "#9fd3a8",
}

#: The clips the figures show, and WHY each one is here — every choice made from
#: the per-clip PIT errors in ``DECOMP_JSON``, never at random. The errors quoted
#: are the 8-mic per-clip means from that file, and ``_per_clip_ranking`` reprints
#: the live ranking next to each figure so a rerun that reshuffles it is visible
#: rather than silent. Between them the six clips cover both rigs and all four
#: regimes, and include the best AND the worst clip of the split.
CLIPS: tuple[dict, ...] = (
    {
        "clip": 35,
        "why": "Michael FLY124 full transition, the ONLY clip in the split that visits "
        "all four regimes (82 frames zero, 81 standby, 48 ramp, 40 cruise) — so "
        "one figure carries the whole decomposition on one time axis. NOTE the "
        "direction: it is a LANDING (cruise -> ramp -> standby -> zero), not a "
        "spin-up; no clip of the frozen split contains a full spin-up to cruise. "
        "It is also the single worst clip in the split for the hard arm (27.22 "
        "rev/s against 16.03 easy), so the four-regime figure is a failure case "
        "and not a showcase.",
    },
    {
        "clip": 22,
        "why": "Michael FLY124 genuine SPIN-UP: rotors barely turning (~5 rev/s, which "
        "the level test calls standby, not zero), then a fast ramp to ~50 rev/s "
        "and a settle at 40. The upward transition the landing clip cannot show, "
        "on the rig the arms were not fitted against; easy 15.55, hard 9.90.",
    },
    {
        "clip": 8,
        "why": "DREGON room1 white-noise TAKE-OFF — zero -> standby -> ramp -> cruise "
        "under a broadband interferer, and the WORST DREGON clip for both arms "
        "(19.90 easy, 21.33 hard). Chosen because it is the failure: the "
        "real-data reference scores 8.15 on the same clip.",
    },
    {
        "clip": 20,
        "why": "DREGON room1 nosource hover — the BEST DREGON clip for the easy arm "
        "(1.98 rev/s, hard 2.98), 251 frames of pure cruise at 80 rev/s. The "
        "honest counterweight to clip 8, and the clip that shows steady cruise "
        "on the fitted rig is essentially solved.",
    },
    {
        "clip": 13,
        "why": "DREGON room1 white-noise hover, also 251 frames of pure cruise — but "
        "the easy arm scores 11.40 here against its own 1.98 on clip 20, and "
        "hard 6.20 against 2.98. Included because it refutes the tidy reading of "
        "the table: the cruise cell is not uniformly solved, it has clip outliers "
        "5-6x its own mean. On MIC 0 the ordering even inverts (easy 3.28, hard "
        "25.14) — a second, separate warning that single-mic views of this split "
        "are not the 8-mic number.",
    },
    {
        "clip": 26,
        "why": "Michael FLY124 cruise with real speed excursions (rotor mean 58 -> 99 "
        "rev/s, 18 ramp frames): cruise on the rig the arms were NOT fitted "
        "against, which is where the transfer claim is actually tested. It is "
        "also the one figure where hard (5.56) is worse than easy (4.79).",
    },
)


plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "text.color": "#111111",
        "axes.labelcolor": "#111111",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    }
)


def spectrogram_db(x: np.ndarray, n: int = 2048, hop: int = 512) -> np.ndarray:
    """``(F, N)`` magnitude spectrogram of one channel, in dB."""
    x = np.asarray(x, dtype=np.float64).ravel()
    w = np.hanning(n + 1)[:n]
    frames = np.stack([x[s : s + n] * w for s in range(0, x.size - n, hop)])
    return 20.0 * np.log10(np.abs(np.fft.rfft(frames, axis=-1)).T + 1e-9)


def segments(labels: np.ndarray) -> list[tuple[str, int, int]]:
    """``labels`` run-length encoded as ``(regime, first, last_exclusive)``."""
    out: list[tuple[str, int, int]] = []
    start = 0
    for i in range(1, labels.size + 1):
        if i == labels.size or labels[i] != labels[start]:
            out.append((str(labels[start]), start, i))
            start = i
    return out


def load_split() -> tuple[list, list[str], list[str]]:
    """The mono (mic 0) dataset, each clip's sample id and its recording id."""
    from data_processing.frame_datasets import DregonLMFrameDataset
    from data_processing.streams import ensure_local

    dataset = DregonLMFrameDataset(
        data_dir=VALID, n_fft=2048, hop_length=512, sample_rate=16000, channel=0
    )
    rows = json.loads(
        (Path(ensure_local(VALID.removeprefix("dload:"))) / "metadata.json").read_text()
    )
    if isinstance(rows, dict):
        rows = next(iter(rows.values()))
    return dataset, [str(r["id"]) for r in rows], [str(r.get("recording_id", "")) for r in rows]


def draw(
    sample_id: str,
    recording: str,
    rig: str,
    audio: np.ndarray,
    target: np.ndarray,
    preds: dict[str, np.ndarray],
    maes: dict[str, float],
    pooled: dict[str, float],
    why: str,
) -> tuple[Path, str]:
    labels = frame_regimes4(target)
    n_frames = target.shape[1]
    duration = audio.size / 16000.0
    t = np.arange(n_frames) / FPS

    fig = plt.figure(figsize=(12.8, 7.4))
    grid = GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[1.0, 0.16, 1.05],
        hspace=0.10,
        left=0.075,
        right=0.995,
        top=0.925,
        bottom=0.175,
    )
    ax_spec = fig.add_subplot(grid[0])
    ax_strip = fig.add_subplot(grid[1], sharex=ax_spec)
    ax_rps = fig.add_subplot(grid[2], sharex=ax_spec)

    spec = spectrogram_db(audio)
    lo, hi = np.percentile(spec, [5, 99.8])
    ax_spec.imshow(
        spec,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0.0, duration, 0.0, 8.0],
        vmin=lo,
        vmax=hi,
        interpolation="nearest",
    )
    ax_spec.set_ylabel("frequency (kHz)")
    ax_spec.set_ylim(0, 8)
    ax_spec.grid(False)
    ax_spec.tick_params(labelbottom=False)
    tags = [tag for _, tag, _ in ARMS]
    mic0 = "  |  ".join(f"{tag} {maes[tag]:.2f}" for tag in tags)
    pooled_note = (
        "\n8-mic PIT MAE   " + "  |  ".join(f"{tag} {pooled[tag]:.2f}" for tag in tags)
        if pooled
        else ""
    )
    ax_spec.set_title(
        f"{rig} · {recording} · {sample_id}\nmic-0 PIT MAE   {mic0}  rev/s{pooled_note}",
        fontsize=13.5,
        pad=7,
        linespacing=1.35,
    )

    # ─ the regime strip, and a boundary rule carried through all three panels
    ax_strip.set_yticks([])
    ax_strip.tick_params(bottom=False, top=False, labelbottom=False)
    ax_strip.grid(False)
    for regime, a, b in segments(labels):
        t0, t1 = a / FPS, b / FPS
        ax_strip.axvspan(t0, t1, color=REGIME_COLOUR[regime], lw=0)
        if (t1 - t0) >= 0.42 * duration / 8:  # room for the word
            ax_strip.text(
                0.5 * (t0 + t1),
                0.5,
                regime,
                ha="center",
                va="center",
                fontsize=11,
                color="#222222",
            )
        if a:
            for ax in (ax_spec, ax_strip, ax_rps):
                ax.axvline(t0, color="#444444", lw=1.0, ls=(0, (4, 3)), alpha=0.75, zorder=5)
    ax_strip.set_ylabel("regime", rotation=0, ha="right", va="center", labelpad=8, fontsize=12)

    # ─ target as one thick light band, every model thin on top. The models are
    # drawn in ARMS order with the reference last so it is never hidden.
    for r in range(target.shape[0]):
        ax_rps.plot(
            t,
            target[r],
            color="#9aa3ad",
            lw=5.0,
            solid_capstyle="round",
            zorder=1,
            label="target (4 rotors)" if r == 0 else None,
        )
    for depth, (_, tag, colour) in enumerate(ARMS):
        emitted = preds[tag]
        for r in range(emitted.shape[0]):
            ax_rps.plot(
                t,
                emitted[r],
                color=colour,
                lw=1.4,
                zorder=3 + depth,
                label=tag if r == 0 else None,
            )
    ax_rps.set_xlabel("time (s)")
    ax_rps.set_ylabel("rotor speed (rev/s)")
    ax_rps.set_xlim(0.0, duration)
    # Three models plus a four-track target is twelve lines; on a cruise-only
    # clip they all live inside a 20 rev/s band, and a fixed 0-105 axis squashes
    # them into one stripe. Fit the axis to the data instead (floored at 0 when
    # the clip visits zero, so a stopped rotor still reads as stopped), with a
    # 30 rev/s minimum span so a very tight clip is not visually exaggerated.
    stack = np.concatenate([target.ravel()] + [p.ravel() for p in preds.values()])
    lo_y, hi_y = float(stack.min()), float(stack.max())
    pad = max(3.0, 0.08 * (hi_y - lo_y))
    lo_y, hi_y = lo_y - pad, hi_y + pad
    if (labels == "zero").any():
        lo_y = min(lo_y, -3.0)
    if hi_y - lo_y < 30.0:
        mid = 0.5 * (lo_y + hi_y)
        lo_y, hi_y = mid - 15.0, mid + 15.0
    ax_rps.set_ylim(lo_y, hi_y)
    # One legend row under the x-label: it can never sit on top of a track, and
    # it names the two arms and every regime the clip actually contains.
    handles, _ = ax_rps.get_legend_handles_labels()
    handles += [
        Patch(color=REGIME_COLOUR[r], label=r, ec="#6b7280", lw=0.7)
        for r in REGIMES
        if (labels == r).any()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.042),
        ncol=len(handles),
        frameon=False,
        fontsize=12,
        handlelength=1.8,
        columnspacing=1.4,
    )
    # The one line that keeps the figure and the table from being confused. It
    # sits below the legend so it can never cover a track.
    fig.text(
        0.53,
        0.004,
        "model tracks drawn RAW, as each model emits them — no matching, so a line "
        "may sit near a target that is not its own rotor;  the MAE above is the "
        "per-frame PIT score",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#444444",
    )

    path = ASSETS / f"pred_real_{rig}_{sample_id}.png"
    fig.savefig(path)
    plt.close(fig)

    shares = {r: int((labels == r).sum()) for r in REGIMES if (labels == r).any()}
    note = (
        f"{path.name}: {rig}/{recording} {sample_id}, {duration:.1f} s, mic 0, 0-8 kHz "
        f"({lo:.0f} to {hi:.0f} dB); regimes {shares}; mic-0 PIT MAE "
        + ", ".join(f"{tag} {maes[tag]:.2f}" for tag in tags)
        + f" rev/s. WHY: {why}"
    )
    return path, note


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    import zoo

    dataset, sample_ids, recordings = load_split()
    models = {}
    for experiment, tag, _ in ARMS:
        model = zoo.load(experiment, ckpt=CKPT, device="cpu")
        salience = bool(getattr(getattr(model, "model", None), "outputs_salience", False))
        models[tag] = (model, salience)

    ranking, pooled_all = _per_clip_ranking()
    notes: list[str] = []
    for spec in CLIPS:
        i = int(spec["clip"])
        frame = dataset[i]
        audio = np.asarray(frame["mixture"].data, dtype=np.float64)
        target = np.asarray(frame["rps"].data, dtype=np.float64)
        preds: dict[str, np.ndarray] = {}
        maes: dict[str, float] = {}
        for _, tag, _c in ARMS:
            model, salience = models[tag]
            pred = predict_clip(model, frame, salience)
            width = min(pred.shape[1], target.shape[1])
            # DRAWN RAW: the model emits four series, so four series are drawn.
            # No permutation, no assignment. PIT belongs to the score only.
            preds[tag] = pred[:, :width]
            maes[tag] = float(pit_abs_error(pred[:, :width], target[:, :width]).mean())
        width = min(min(p.shape[1] for p in preds.values()), target.shape[1])
        rig = (
            "michaels"
            if "michael" in recordings[i].lower() or recordings[i].upper().startswith("FLY")
            else "dregon"
        )
        path, note = draw(
            sample_ids[i],
            recordings[i],
            rig,
            audio,
            target[:, :width],
            {k: v[:, :width] for k, v in preds.items()},
            maes,
            pooled_all.get(i, {}),
            str(spec["why"]),
        )
        if i in ranking:
            note += f" [8-mic per-clip rank: {ranking[i]}]"
        notes.append(note)
        print(note, flush=True)

    print(f"\nwrote {len(notes)} figures to {ASSETS} in {time.time() - t0:.1f} s:")
    for note in notes:
        name = note.split(":", 1)[0]
        p = ASSETS / name
        print(f"  {p.relative_to(ROOT)}  {p.stat().st_size / 1e3:.0f} kB")
    return 0


def _per_clip_ranking() -> tuple[dict[int, str], dict[int, dict[str, float]]]:
    """Per-clip rank notes and pooled 8-mic MAEs from the decomposition JSON.

    Read-only. The ranks annotate the printed figure list so the clip choices
    can be checked against the numbers that motivated them, per rig (the rigs
    are not comparable, so a global ranking would be meaningless). The pooled
    MAEs go into each figure title next to its mic-0 value, because one mic can
    sit far from the 8-mic mean and a title showing only mic 0 would mislead.
    """
    if not DECOMP_JSON.is_file():
        return {}, {}
    rows = {r["experiment"]: r for r in json.loads(DECOMP_JSON.read_text())["rows"]}
    out: dict[int, list[str]] = {}
    pooled: dict[int, dict[str, float]] = {}
    for experiment, tag, _ in ARMS:
        row = rows.get(experiment)
        if row is None:
            continue
        for rig in ("dregon", "michaels"):
            clips = [
                c for c in row["per_clip"] if c["rig"] == rig and c["overall"]["mae"] is not None
            ]
            order = sorted(clips, key=lambda c: -c["overall"]["mae"])
            for k, c in enumerate(order, start=1):
                out.setdefault(int(c["clip"]), []).append(
                    f"{tag} worst #{k}/{len(order)} in {rig} ({c['overall']['mae']:.2f})"
                )
                pooled.setdefault(int(c["clip"]), {})[tag] = float(c["overall"]["mae"])
    return {k: "; ".join(v) for k, v in out.items()}, pooled


if __name__ == "__main__":
    raise SystemExit(main())
