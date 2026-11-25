"""Arrange finished November cruise-generator outputs into page assets.

Reads only ``results/november_cruise_generator/<FLY>/`` (the fit/render CLI's
report contract). It fits nothing and renders no model audio. Every number on
the page is computed here from the saved, unscaled held-out WAV files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = HERE / "assets"

# Condition labels are fixed by the experiment design, not read from the fit.
RECORDINGS = {
    "FLY013": ("Bare-motor running", "bare-motor running, not airborne flight"),
    "FLY018": ("Airborne cruise", "airborne cruise"),
    "FLY022": ("Airborne cruise", "airborne cruise"),
}
EXAMPLES = (1, 2)
CHANNELS = ("ch 1", "ch 2")

PEAK_TARGET_DBFS = -1.0  # playback copies: louder peak of a pair lands here when a gain is needed
QUIET_PEAK_DBFS = -20.0  # below this pair peak, raise the pair for audibility
SPEC_FRAME_S = 0.128  # STFT frame length (2048 samples at 16 kHz)
SPEC_HOP_DIV = 8  # hop = frame / 8
SPEC_RANGE_DB = 70.0
WELCH_FRAME_S = 0.512  # Welch segment (8192 samples at 16 kHz)
BAND_EDGE_FRACTION = 0.49  # keep 1/3-octave bands whose upper edge <= 0.49 sr
LOWEST_BAND_N = -13  # 1000 * 10^(-13/10) = 50.1 Hz
F_MIN_PLOT = 40.0

REC_COLOR, GEN_COLOR, NATIVE_COLOR = "#1f2933", "#d9480f", "#8a94a6"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_stereo(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    if x.shape[1] != 2:
        raise ValueError(f"{path}: expected 2 channels, found {x.shape[1]}")
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: non-finite samples")
    return x.T.copy(), int(sr)


def db(x: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(x, 1e-30))


def peak_dbfs(x: np.ndarray) -> float:
    return float(20.0 * np.log10(max(float(np.abs(x).max()), 1e-15)))


def rms_db(x: np.ndarray) -> list[float]:
    return [float(db(np.mean(c**2))) for c in x]


def portable(value):
    """Drop host paths from metadata strings before they reach the page."""
    if isinstance(value, dict):
        return {str(k): portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str) and (value.startswith(("/", "~")) or "/home/" in value):
        return f"…/{Path(value).name}"
    return value


def flatten(value, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        rows: list[tuple[str, str]] = []
        for k, v in value.items():
            rows += flatten(v, f"{prefix}.{k}" if prefix else str(k))
        return rows
    return [(prefix or "value", brief(value))]


def brief(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        if len(value) <= 8 and all(not isinstance(v, (dict, list)) for v in value):
            return "[" + ", ".join(brief(v) for v in value) + "]"
        return f"list of {len(value)}"
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in flatten(value))
    return str(value)


def cell(text: str) -> str:
    return escape(text).replace("|", "\\|")


# --------------------------------------------------------------------- spectra


def welch(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(WELCH_FRAME_S * sr))
    return signal.welch(
        x,
        fs=sr,
        window="hann",
        nperseg=n,
        noverlap=n // 2,
        detrend="constant",
        scaling="density",
        axis=-1,
    )


def third_octave_bands(sr: int) -> np.ndarray:
    rows, n = [], LOWEST_BAND_N
    while True:
        fc = 1000.0 * 10.0 ** (n / 10.0)
        lo, hi = fc * 10.0 ** (-0.05), fc * 10.0**0.05
        if hi > BAND_EDGE_FRACTION * sr:
            break
        rows.append((fc, lo, hi))
        n += 1
    return np.asarray(rows)


def band_power(f: np.ndarray, psd: np.ndarray, bands: np.ndarray) -> np.ndarray:
    df = f[1] - f[0]
    out = np.empty(psd.shape[:-1] + (len(bands),))
    for b, (_, lo, hi) in enumerate(bands):
        sel = (f >= lo) & (f < hi)
        if not sel.any():
            raise ValueError(f"band {lo:.1f}-{hi:.1f} Hz has no Welch bin")
        out[..., b] = psd[..., sel].sum(-1) * df
    return out


def pair_metrics(ref: np.ndarray, gen: np.ndarray, sr: int) -> dict:
    f, pr = welch(ref, sr)
    _, pg = welch(gen, sr)
    bands = third_octave_bands(sr)
    br, bg = band_power(f, pr, bands), band_power(f, pg, bands)
    d = db(bg) - db(br)  # (2, B)
    bias = float(d.mean())
    return {
        "recorded_rms_db": rms_db(ref),
        "generated_rms_db": rms_db(gen),
        "level_error_db": [float(db(bg[c].sum()) - db(br[c].sum())) for c in range(2)],
        "band_distance_db": float(np.sqrt(np.mean(d**2))),
        "band_offset_db": bias,
        "band_shape_db": float(np.sqrt(np.mean((d - bias) ** 2))),
        "band_centres_hz": bands[:, 0].tolist(),
        "band_range_hz": [float(bands[0, 1]), float(bands[-1, 2])],
        "band_difference_db": d.tolist(),
    }


def spectrogram_db(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(round(SPEC_FRAME_S * sr))
    hop = n // SPEC_HOP_DIV
    f, t, s = signal.spectrogram(
        x,
        fs=sr,
        window="hann",
        nperseg=n,
        noverlap=n - hop,
        detrend=False,
        scaling="spectrum",
        mode="psd",
        axis=-1,
    )
    return f, t, db(s.mean(0))


def short_time_level(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(SPEC_FRAME_S * sr))
    hop = n // SPEC_HOP_DIV
    frames = np.lib.stride_tricks.sliding_window_view(x, n, axis=-1)[:, ::hop]
    level = db(np.mean(frames**2, axis=(0, 2)))
    return (np.arange(level.size) * hop + n / 2) / sr, level


# ---------------------------------------------------------------- one flight


def load_flight(results: Path, inputs: Path, fly: str) -> dict:
    d = results / fly
    report = json.loads((d / "report.json").read_text())
    missing = [
        k
        for k in (
            "recording",
            "sr",
            "seed",
            "fit_start_s",
            "fit_duration_s",
            "examples",
            "model",
            "fit",
            "limitations",
        )
        if k not in report
    ]
    if missing:
        raise KeyError(f"{d / 'report.json'} lacks {missing}")
    if report["recording"] != fly:
        raise ValueError(f"{d}: report.json names recording {report['recording']!r}")
    sr = int(report["sr"])
    examples = {int(e["index"]): e for e in report["examples"]}
    if sorted(examples) != list(EXAMPLES):
        raise ValueError(f"{d}: examples must be indexed 1 and 2, found {sorted(examples)}")
    fit_start = [float(s) for s in report["fit_start_s"]]
    dur = report["fit_duration_s"]
    fit_dur = [float(v) for v in dur] if isinstance(dur, list) else [float(dur)] * len(fit_start)
    if len(fit_dur) != len(fit_start):
        raise ValueError(f"{d}: {len(fit_start)} fit starts but {len(fit_dur)} durations")

    pairs = {}
    for i in EXAMPLES:
        ref, sr_r = load_stereo(d / f"reference_{i}.wav")
        gen, sr_g = load_stereo(d / f"generated_{i}.wav")
        if sr_r != sr or sr_g != sr:
            raise ValueError(
                f"{d}: example {i} sample rates {sr_r}/{sr_g} differ from report sr {sr}"
            )
        if ref.shape != gen.shape:
            raise ValueError(f"{d}: example {i} shapes {ref.shape} vs {gen.shape}")
        n_expected = float(examples[i]["duration_s"]) * sr
        if abs(ref.shape[1] - n_expected) > 1:
            raise ValueError(
                f"{d}: example {i} has {ref.shape[1]} samples, report says {n_expected:.0f}"
            )
        native_path = d / f"native_reference_{i}.wav"
        native = None
        if native_path.exists():
            nx, nsr = load_stereo(native_path)
            if abs(nx.shape[1] / nsr - ref.shape[1] / sr) > 1.0 / sr:
                raise ValueError(f"{native_path}: duration differs from reference_{i}.wav")
            native = {"x": nx, "sr": nsr, "path": native_path}
        pairs[i] = {"ref": ref, "gen": gen, "native": native, **examples[i]}

    # Disjointness on the recording clock: fit windows vs held-out examples and the two examples.
    fit_iv = [(s, s + w) for s, w in zip(fit_start, fit_dur)]
    ex_iv = [
        (float(pairs[i]["start_s"]), float(pairs[i]["start_s"]) + float(pairs[i]["duration_s"]))
        for i in EXAMPLES
    ]

    def gap(a, b):
        return max(a[0], b[0]) - min(a[1], b[1])

    fit_gap = min(gap(a, b) for a in fit_iv for b in ex_iv)
    ex_gap = gap(*ex_iv)
    if fit_gap <= 0 or ex_gap <= 0:
        raise ValueError(
            f"{fly}: windows overlap (fit-to-example gap {fit_gap:.3f} s, example gap {ex_gap:.3f} s)"
        )

    selection = report.get("selection")
    for candidate in (d / "selection.json", inputs / fly / "selection.json"):
        if selection is None and candidate.exists():
            selection = json.loads(candidate.read_text())
    if selection is not None:
        # The page states the parent's selection; it must describe the windows the fit actually used.
        if selection.get("recording", fly) != fly:
            raise ValueError(f"{fly}: selection record names {selection.get('recording')!r}")

        def same(a, b) -> bool:
            a, b = sorted(map(float, a)), sorted(map(float, b))
            return len(a) == len(b) and np.allclose(a, b, rtol=0, atol=1e-3)

        if "fit_start_s" in selection and not same(selection["fit_start_s"], fit_start):
            raise ValueError(f"{fly}: fit starts differ between selection.json and report.json")
        if "heldout_start_s" in selection and not same(
            selection["heldout_start_s"], [a for a, _ in ex_iv]
        ):
            raise ValueError(
                f"{fly}: held-out starts differ between selection.json and report.json"
            )

    sources = sorted(p for p in d.iterdir() if p.suffix in (".wav", ".json"))
    return {
        "fly": fly,
        "dir": d,
        "report": report,
        "sr": sr,
        "fit_iv": fit_iv,
        "ex_iv": ex_iv,
        "fit_gap_s": fit_gap,
        "example_gap_s": ex_gap,
        "pairs": pairs,
        "selection": portable(selection) if selection is not None else None,
        "hashes": {p.name: sha256(p) for p in sources},
    }


def playback_gain_db(pair: dict) -> float:
    waves = [pair["ref"], pair["gen"]] + ([pair["native"]["x"]] if pair["native"] else [])
    peak = max(peak_dbfs(w) for w in waves)
    if peak > PEAK_TARGET_DBFS or peak < QUIET_PEAK_DBFS:
        return PEAK_TARGET_DBFS - peak
    return 0.0


def write_playback(path: Path, x: np.ndarray, sr: int, gain_db: float) -> None:
    y = (x * 10.0 ** (gain_db / 20.0)).astype(np.float32)
    assert float(np.abs(y).max()) <= 1.0
    sf.write(path, y.T, sr, subtype="FLOAT")


def _save_spectrogram_views(fig, freq_axis, out: Path) -> None:
    # Reuse the same plotted mesh and colour limits; only its y transform changes.
    for mode, path in (("log", out), ("linear", out.with_stem(out.stem + "-linear"))):
        freq_axis.set_yscale(mode)
        freq_axis.set_ylabel(f"Frequency (Hz, {mode})")
        fig.savefig(path, dpi=150)
    plt.close(fig)


def spectrogram_figure(
    fly: str, i: int, pair: dict, sr: int, scale: tuple[float, float], out: Path
) -> dict:
    vmin, vmax = scale
    fig = plt.figure(figsize=(11.5, 5.4), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(3.2, 1.0))
    axes = [fig.add_subplot(grid[0, 0]), None]
    axes[1] = fig.add_subplot(grid[0, 1], sharex=axes[0], sharey=axes[0])
    saturated = {}
    for ax, key, title in ((axes[0], "ref", "Recorded"), (axes[1], "gen", "Generated (seed draw)")):
        f, t, s = spectrogram_db(pair[key], sr)
        keep = f >= F_MIN_PLOT
        mesh = ax.pcolormesh(
            t, f[keep], s[keep], cmap="magma", vmin=vmin, vmax=vmax, shading="auto", rasterized=True
        )
        saturated[key] = float(np.mean(s[keep] > vmax))
        ax.set_ylim(F_MIN_PLOT, sr / 2)
        ax.set_title(f"{title} · {sr / 1000:g} kHz, mean of both channels", fontsize=11, loc="left")
        ax.set_xlabel("Time in example (s)")
    plt.setp(axes[1].get_yticklabels(), visible=False)
    bar = fig.colorbar(mesh, ax=axes, pad=0.01, fraction=0.03)
    bar.set_label("Power spectrum (dB re full scale²)")
    lev = fig.add_subplot(grid[1, :])
    for key, color, label in (("ref", REC_COLOR, "Recorded"), ("gen", GEN_COLOR, "Generated")):
        t, level = short_time_level(pair[key], sr)
        lev.plot(t, level, color=color, lw=1.4, label=label)
    lev.set_xlim(0, pair["ref"].shape[1] / sr)
    lev.set_ylabel("Level (dB)")
    lev.set_xlabel("Time in example (s)")
    lev.grid(alpha=0.3)
    lev.legend(loc="lower right", ncols=2, fontsize=9, frameon=False)
    fig.suptitle(f"{fly} · held-out example {i}", fontsize=13, x=0.01, ha="left")
    _save_spectrogram_views(fig, axes[0], out)  # both panels share y
    return saturated


def native_figure(
    fly: str, i: int, pair: dict, sr_model: int, scale: tuple[float, float], out: Path
) -> None:
    nat = pair["native"]
    f, t, s = spectrogram_db(nat["x"], nat["sr"])
    keep = f >= F_MIN_PLOT
    fig, ax = plt.subplots(figsize=(11.5, 3.6), layout="constrained")
    mesh = ax.pcolormesh(
        t,
        f[keep],
        s[keep],
        cmap="magma",
        vmin=scale[0],
        vmax=scale[1],
        shading="auto",
        rasterized=True,
    )
    ax.axhline(sr_model / 2, color="#7fdbff", lw=1.4, ls="--")
    ax.text(
        t[-1],
        sr_model / 2 * 1.06,
        f"model output Nyquist {sr_model / 2000:g} kHz",
        color="#7fdbff",
        ha="right",
        va="bottom",
        fontsize=9,
    )
    ax.set_ylim(F_MIN_PLOT, nat["sr"] / 2)
    ax.set_xlabel("Time in example (s)")
    ax.set_title(
        f"{fly} · example {i} · recorded at native {nat['sr'] / 1000:g} kHz (outside the comparison)",
        fontsize=11,
        loc="left",
    )
    fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.03).set_label("dB re full scale²")
    _save_spectrogram_views(fig, ax, out)


def psd_figure(flight: dict, metrics: dict, out: Path) -> None:
    sr = flight["sr"]
    natives = [p["native"] for p in flight["pairs"].values() if p["native"]]
    f_top = max([sr / 2] + [n["sr"] / 2 for n in natives])
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 6.4),
        sharex="col",
        height_ratios=(2.2, 1.0),
        layout="constrained",
        squeeze=False,
    )
    for col, i in enumerate(EXAMPLES):
        pair, m = flight["pairs"][i], metrics[i]
        ax, axd = axes[0, col], axes[1, col]
        for key, color, label, lw in (
            ("ref", REC_COLOR, f"Recorded ({sr / 1000:g} kHz)", 1.3),
            ("gen", GEN_COLOR, "Generated", 1.3),
        ):
            f, p = welch(pair[key], sr)
            ax.plot(f[1:], db(p.mean(0))[1:], color=color, lw=lw, label=label)
        if pair["native"]:
            f, p = welch(pair["native"]["x"], pair["native"]["sr"])
            ax.plot(
                f[1:],
                db(p.mean(0))[1:],
                color=NATIVE_COLOR,
                lw=0.8,
                alpha=0.9,
                zorder=0,
                label=f"Recorded, native {pair['native']['sr'] / 1000:g} kHz (not scored)",
            )
        lo, hi = m["band_range_hz"]
        for a in (ax, axd):
            a.axvspan(lo, hi, color="#2f9e44", alpha=0.06, lw=0)
            if f_top > sr / 2:
                a.axvspan(sr / 2, f_top, color="#868e96", alpha=0.12, lw=0)
            a.set_xscale("log")
            a.set_xlim(20, f_top)
            a.grid(alpha=0.3, which="both", lw=0.5)
        ax.set_title(
            f"Example {i} · {pair['start_s']:.1f}–{pair['start_s'] + pair['duration_s']:.1f} s",
            loc="left",
            fontsize=11,
        )
        ax.set_ylabel("PSD, mean of channels (dB/Hz)")
        ax.legend(fontsize=8.5, loc="lower left", frameon=False)
        d = np.asarray(m["band_difference_db"])
        fc = np.asarray(m["band_centres_hz"])
        axd.axhline(0, color="k", lw=0.8)
        axd.plot(fc, d[0], "o-", ms=3.5, lw=1, color="#1971c2", label=CHANNELS[0])
        axd.plot(fc, d[1], "s--", ms=3.5, lw=1, color="#e8590c", label=CHANNELS[1])
        axd.set_ylabel("Generated − recorded\n1/3-octave band (dB)")
        axd.set_xlabel("Frequency (Hz)")
        axd.legend(fontsize=8.5, ncols=2, loc="best", frameon=False)
    fig.suptitle(
        f"{flight['fly']} · Welch PSD of the unscaled held-out audio (green: scored band range; grey: above the model's Nyquist)",
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.savefig(out, dpi=150)
    plt.close(fig)


def timeline_figure(flights: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(
        len(flights),
        1,
        figsize=(11.5, 1.35 * len(flights) + 0.7),
        layout="constrained",
        squeeze=False,
    )
    for ax, fl in zip(axes[:, 0], flights):
        ax.broken_barh(
            [(a, b - a) for a, b in fl["fit_iv"]],
            (0.55, 0.35),
            color="#1971c2",
            label="Fit (calibration) windows",
        )
        ax.broken_barh(
            [(a, b - a) for a, b in fl["ex_iv"]],
            (0.1, 0.35),
            color=GEN_COLOR,
            label="Held-out examples",
        )
        for i, (a, b) in zip(EXAMPLES, fl["ex_iv"]):
            ax.text(
                (a + b) / 2,
                0.275,
                str(i),
                color="white",
                ha="center",
                va="center",
                fontsize=9,
                weight="bold",
            )
        ends = [v for iv in fl["fit_iv"] + fl["ex_iv"] for v in iv]
        pad = 0.03 * (max(ends) - min(ends))
        ax.set_xlim(min(ends) - pad, max(ends) + pad)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(fl["fly"], rotation=0, ha="right", va="center", fontsize=10)
        ax.text(
            1.0,
            1.02,
            f"closest fit–held-out gap {fl['fit_gap_s']:.1f} s",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
        )
        ax.grid(axis="x", alpha=0.3)
    axes[-1, 0].set_xlabel("Recording clock (s)")
    axes[0, 0].legend(
        loc="lower left", bbox_to_anchor=(0, 1.02), ncols=2, fontsize=9, frameon=False
    )
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ markdown


def fmt_pair(v: list[float]) -> str:
    return " / ".join(f"{x:+.1f}" for x in v)


def gain_label(g: float) -> str:
    return "0.0 dB (raw digital level)" if g == 0.0 else f"{g:+.1f} dB"


def player_html(fly: str, i: int, pair: dict, gain: float, sr: int) -> str:
    stem = f"audio/{fly}_{i}"
    a = f"assets/{stem}_recorded.wav"
    b = f"assets/{stem}_generated.wav"
    window = f"{pair['start_s']:.1f}–{pair['start_s'] + pair['duration_s']:.1f} s"
    html = [
        "```{=html}",
        f'<div class="ab-player" data-ab aria-label="{escape(fly)} example {i} A/B comparison">',
        f'<div class="ab-head"><strong>A/B · example {i}</strong><span>held-out {window} · '
        f"{sr / 1000:g} kHz stereo · playback gain on both files: <strong>{escape(gain_label(gain))}</strong></span></div>",
        '<div class="ab-controls" hidden>',
        '<button type="button" class="ab-play" data-ab-play aria-pressed="false">▶ Play</button>',
        '<div class="ab-switch" role="group" aria-label="Choose source">',
        '<button type="button" data-ab-pick="a" aria-pressed="true">A · Recorded</button>',
        '<button type="button" data-ab-pick="b" aria-pressed="false">B · Generated</button>',
        "</div>",
        '<label class="ab-loop"><input type="checkbox" data-ab-loop checked> Loop</label>',
        '<span class="ab-time" data-ab-time aria-live="off">0.00 s</span>',
        "</div>",
        '<div class="ab-tracks">',
    ]
    for key, src, label in (("a", a, "A · Recorded"), ("b", b, "B · Generated")):
        html += [
            f'<figure class="ab-track" data-ab-track="{key}"><figcaption>{label}</figcaption>',
            f'<audio controls preload="auto" aria-label="{escape(fly)} example {i} {label}">'
            f'<source src="{src}" type="audio/wav"><a href="{src}">Download WAV</a></audio></figure>',
        ]
    html += ["</div>", "</div>"]
    if pair["native"]:
        nsr = pair["native"]["sr"]
        src = f"assets/{stem}_native.wav"
        html += [
            '<div class="native-ref">',
            f"<p><strong>Outside the A/B: recorded at native {nsr / 1000:g} kHz.</strong> Same window and the same "
            f"playback gain ({escape(gain_label(gain))}). It contains content above {sr / 2000:g} kHz that the "
            f"{sr / 1000:g} kHz model output cannot contain. It is not scored and is not a fair comparison partner.</p>",
            f'<audio controls preload="metadata" aria-label="{escape(fly)} example {i} native recording">'
            f'<source src="{src}" type="audio/wav"><a href="{src}">Download WAV</a></audio>',
            "</div>",
        ]
    html.append("```")
    return "\n".join(html)


def metadata_table(rows: list[tuple[str, str]]) -> list[str]:
    out = ["| Setting | Value |", "|---|---|"]
    out += [f"| `{cell(k)}` | {cell(v)} |" for k, v in rows]
    return out


def heldout_window(selection: dict | None, start: float) -> dict | None:
    if not selection:
        return None
    for w in selection.get("windows", []):
        if w.get("split") == "holdout" and abs(float(w.get("start_s", -1e9)) - start) < 1e-3:
            return w
    return None


def _spectrogram_markup(stem: str, caption: str) -> str:
    images = [
        f'<img data-freq-view="{mode}" src="assets/fig/{stem}{suffix}.png" '
        f'alt="{escape(caption)} Frequency axis: {mode}.">'
        for mode, suffix in (("log", ""), ("linear", "-linear"))
    ]
    return "\n".join(
        [
            "```{=html}",
            '<figure class="spectrogram-views">',
            *images,
            f"<figcaption>{escape(caption)}</figcaption>",
            "</figure>",
            "```",
        ]
    )


def flight_markdown(fl: dict, metrics: dict, gains: dict, saturation: dict) -> str:
    fly, rep, sr = fl["fly"], fl["report"], fl["sr"]
    label, condition = RECORDINGS[fly]
    lines = [f"### {fly} · {label}", ""]
    lines += [
        f"{condition.capitalize()} · {sr / 1000:g} kHz stereo · one draw per excerpt "
        f"(master seed {rep['seed']}). Fit and listening windows are at least "
        f"{fl['fit_gap_s']:.0f} seconds apart.",
        "",
    ]
    details = [
        "| Example | Window (s) | Recorded RMS (dB) ch 1 / ch 2 | Generated RMS (dB) ch 1 / ch 2 | Level error (dB) ch 1 / ch 2 | Band distance (dB) | Band offset (dB) | Shape distance (dB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i in EXAMPLES:
        m, (a, b) = metrics[i], fl["ex_iv"][i - 1]
        details.append(
            f"| {i} | {a:g}–{b:g} | {fmt_pair(m['recorded_rms_db'])} | {fmt_pair(m['generated_rms_db'])} | "
            f"{fmt_pair(m['level_error_db'])} | {m['band_distance_db']:.1f} | {m['band_offset_db']:+.1f} | "
            f"{m['band_shape_db']:.1f} |"
        )
    lo, hi = metrics[1]["band_range_hz"]
    details += [
        "",
        f": RMS is of the unscaled samples (0 dB = RMS 1.0). Band metrics use {len(metrics[1]['band_centres_hz'])} "
        f"one-third-octave bands, {lo:.0f} Hz – {hi:.0f} Hz. {{.striped}}",
        "",
    ]
    for i in EXAMPLES:
        pair = fl["pairs"][i]
        title = (
            {1: "High-speed plateau", 2: "Low-speed running"}[i]
            if fly == "FLY013"
            else f"Cruise excerpt {i}"
        )
        lines += [
            f"#### {title} · {pair['start_s']:g}–{pair['start_s'] + pair['duration_s']:g} s",
            "",
        ]
        speeds = heldout_window(fl["selection"], float(pair["start_s"]))
        if fly == "FLY013" and i == 2:
            lines += [
                "Around **3.8 seconds into this excerpt**, the recorded motor tones change "
                "frequency. The generated upper tones move at the same time, but several "
                "lower tones are much weaker. Switch A/B here to hear the difference in "
                "spectral balance even though the overall levels are close.",
                "",
            ]
        if speeds is not None and speeds.get("mean_rotor_rps"):
            lines += [
                "Mean rotor speeds (right front / left front / left rear / right rear): "
                + " / ".join(f"{v:.1f}" for v in speeds["mean_rotor_rps"])
                + f" rev/s; range {speeds['min_rps']:.1f}–{speeds['max_rps']:.1f} rev/s.",
                "",
            ]
        lines += [player_html(fly, i, pair, gains[i], sr), ""]
        lines += [
            "```{=html}",
            '<label class="freq-control" hidden>Spectrogram frequency axis '
            '<select data-freq-select><option value="log" selected>Logarithmic</option>'
            '<option value="linear">Linear</option></select></label>',
            "```",
            "",
            _spectrogram_markup(
                f"{fly}_{i}_spectrogram",
                f"{fly} example {i}: recorded (left) and generated (right) on one colour scale, "
                "with short-time level below.",
            ),
            "",
        ]
        sat = saturation[i]
        if max(sat.values()) > 0.01:
            lines += [
                f"Above the colour scale: recorded {100 * sat['ref']:.1f} %, generated {100 * sat['gen']:.1f} % of "
                "time–frequency cells (they show at the top colour).",
                "",
            ]
        if pair["native"]:
            lines += [
                _spectrogram_markup(
                    f"{fly}_{i}_native",
                    f"{fly} example {i}: native-rate recording. The dashed line is the "
                    "model output Nyquist frequency.",
                ),
                "",
            ]
    lines += ["#### Held-out levels and spectral errors", ""] + details
    lines += [
        "#### Spectra of both examples",
        "",
        f"![{fly}: Welch PSD of recorded and generated held-out audio (top) and the per-band difference (bottom).]"
        f'(assets/fig/{fly}_psd.png){{fig-alt="Welch power spectral densities and one-third-octave band differences '
        f'for {fly}."}}',
        "",
        "#### Fit settings and limitations",
        "",
        "Reported by the fit:",
        "",
    ]
    lines += [f"- {cell(str(x))}" for x in portable(rep["limitations"])] or ["- (none reported)"]
    lines += [
        "",
        "<details><summary>Optimizer, objective, configuration and convergence</summary>",
        "",
    ]
    fit = portable(rep["fit"])
    lines += metadata_table(flatten(fit))
    lines += ["", "</details>", ""]
    if fl["selection"] is not None:
        sel = fl["selection"]
        lines += ["<details><summary>Window selection record</summary>", ""]
        lines += metadata_table(flatten({k: v for k, v in sel.items() if k != "windows"}))
        if sel.get("windows"):
            lines += [
                "",
                "| Split | Window (s) | Mean speed per rotor (rev/s) | Min–max speed (rev/s) | Longest speed-packet gap (s) |",
                "|---|---|---|---|---|",
            ]
            for w in sorted(
                sel["windows"], key=lambda w: (w.get("split") != "fit", w.get("start_s", 0))
            ):
                lines.append(
                    f"| {cell(str(w.get('split', '')))} | {w.get('start_s', float('nan')):g}–{w.get('end_s', float('nan')):g} | "
                    + " / ".join(f"{v:.1f}" for v in w.get("mean_rotor_rps", []))
                    + f" | {w.get('min_rps', float('nan')):.1f}–{w.get('max_rps', float('nan')):.1f} | "
                    + f"{w.get('max_packet_gap_s', float('nan')):.3f} |"
                )
        lines += ["", "</details>", ""]
    return "\n".join(lines)


def convergence_text(fit: dict) -> str:
    conv = fit["convergence"]
    if conv["converged"]:
        return "Converged"
    reason = {"deadline": "time limit", "max_restart_passes": "restart limit"}.get(
        conv["stop"], conv["stop"]
    )
    return f"Stopped at {reason}; not converged"


def summary_markdown(flights: list[dict], metrics: dict) -> str:
    lines = [
        "| Recording | Condition | Seed | Fit windows | Held-out (s) | Level error (dB), ex. 1 · ex. 2, ch 1 / ch 2 | Band distance (dB), ex. 1 · ex. 2 | Optimizer |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for fl in flights:
        m = metrics[fl["fly"]]
        durations = " / ".join(f"{d:g}" for d in sorted({b - a for a, b in fl["fit_iv"]}))
        lines.append(
            f"| **{fl['fly']}** | {RECORDINGS[fl['fly']][0]} | {fl['report']['seed']} | "
            f"{len(fl['fit_iv'])} × {durations} s | "
            + ", ".join(f"{a:g}–{b:g}" for a, b in fl["ex_iv"])
            + " | "
            + " · ".join(fmt_pair(m[i]["level_error_db"]) for i in EXAMPLES)
            + " | "
            + " · ".join(f"{m[i]['band_distance_db']:.1f}" for i in EXAMPLES)
            + f" | {cell(convergence_text(fl['report']['fit']))} |"
        )
    return "\n".join(lines) + "\n"


def provenance_markdown(flights: list[dict], frames: set[str]) -> str:
    lines = []
    if frames:
        lines += ["Data frames: " + ", ".join(f"`{cell(f)}`" for f in sorted(frames)) + ".", ""]
    lines += ["| Recording | File | SHA-256 (first 16 hex) |", "|---|---|---|"]
    for fl in flights:
        for name, h in fl["hashes"].items():
            lines.append(f"| {fl['fly']} | `{name}` | `{h[:16]}` |")
    return "\n".join(lines) + "\n"


def find_frames(value) -> set[str]:
    found = set()
    if isinstance(value, dict):
        if isinstance(value.get("dataset"), str) and isinstance(value.get("version"), str):
            found.add(f"{value['dataset']}@{value['version']}")
        for v in value.values():
            found |= find_frames(v)
    elif isinstance(value, list):
        for v in value:
            found |= find_frames(v)
    elif isinstance(value, str) and "-frames@" in value:
        found.add(value.rsplit("/", 1)[-1])
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/november_cruise_generator")
    parser.add_argument(
        "--inputs",
        type=Path,
        default=None,
        help="fit input directory holding <FLY>/selection.json (default RESULTS/inputs)",
    )
    parser.add_argument(
        "--recordings", nargs="+", default=list(RECORDINGS), choices=list(RECORDINGS)
    )
    args = parser.parse_args()

    flights = [
        load_flight(args.results, args.inputs or args.results / "inputs", fly)
        for fly in args.recordings
    ]
    if ASSETS.exists():
        shutil.rmtree(ASSETS)
    (ASSETS / "audio").mkdir(parents=True)
    (ASSETS / "fig").mkdir()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    all_metrics, media, frames = {}, {}, set()
    flight_pages = ["::: {.panel-tabset}", ""]
    for fl in flights:
        fly, sr = fl["fly"], fl["sr"]
        frames |= find_frames(fl["report"]) | find_frames(fl["selection"])
        metrics = {
            i: pair_metrics(fl["pairs"][i]["ref"], fl["pairs"][i]["gen"], sr) for i in EXAMPLES
        }
        all_metrics[fly] = metrics
        # One colour scale per recording, anchored to the 99.5th percentile of all four 16 kHz panels.
        top = max(
            float(np.percentile(spectrogram_db(fl["pairs"][i][k], sr)[2], 99.5))
            for i in EXAMPLES
            for k in ("ref", "gen")
        )
        scale = (math.ceil(top + 3.0) - SPEC_RANGE_DB, math.ceil(top + 3.0))
        gains, saturation = {}, {}
        for i in EXAMPLES:
            pair = fl["pairs"][i]
            gains[i] = playback_gain_db(pair)
            for key, name in (("ref", "recorded"), ("gen", "generated")):
                write_playback(ASSETS / f"audio/{fly}_{i}_{name}.wav", pair[key], sr, gains[i])
            if pair["native"]:
                write_playback(
                    ASSETS / f"audio/{fly}_{i}_native.wav",
                    pair["native"]["x"],
                    pair["native"]["sr"],
                    gains[i],
                )
                native_figure(fly, i, pair, sr, scale, ASSETS / f"fig/{fly}_{i}_native.png")
            saturation[i] = spectrogram_figure(
                fly, i, pair, sr, scale, ASSETS / f"fig/{fly}_{i}_spectrogram.png"
            )
        psd_figure(fl, metrics, ASSETS / f"fig/{fly}_psd.png")
        flight_pages += [flight_markdown(fl, metrics, gains, saturation), ""]
        media[fly] = {
            "sr": sr,
            "seed": fl["report"]["seed"],
            "spectrogram_scale_db": list(scale),
            "fit_gap_s": fl["fit_gap_s"],
            "example_gap_s": fl["example_gap_s"],
            "examples": {
                str(i): {
                    "playback_gain_db": gains[i],
                    "raw_peak_dbfs": {
                        "recorded": peak_dbfs(fl["pairs"][i]["ref"]),
                        "generated": peak_dbfs(fl["pairs"][i]["gen"]),
                        **(
                            {"native": peak_dbfs(fl["pairs"][i]["native"]["x"])}
                            if fl["pairs"][i]["native"]
                            else {}
                        ),
                    },
                    "native_sr": fl["pairs"][i]["native"]["sr"]
                    if fl["pairs"][i]["native"]
                    else None,
                    "metrics": {k: v for k, v in metrics[i].items() if k != "band_difference_db"},
                }
                for i in EXAMPLES
            },
            "source_sha256": fl["hashes"],
        }
    flight_pages.append(":::")
    timeline_figure(flights, ASSETS / "fig/timeline.png")
    (ASSETS / "flights.md").write_text("\n".join(flight_pages) + "\n")
    (ASSETS / "summary.md").write_text(summary_markdown(flights, all_metrics))
    (ASSETS / "provenance.md").write_text(provenance_markdown(flights, frames))
    (ASSETS / "media.json").write_text(json.dumps(media, indent=2) + "\n")
    n_native = sum(1 for fl in flights for p in fl["pairs"].values() if p["native"])
    print(
        f"prepared {len(flights)} recordings, {2 * len(flights)} A/B pairs, {n_native} native references → {ASSETS}"
    )


if __name__ == "__main__":
    main()
