"""Build the assets of ``docs/explainers/diverse-comb.qmd``: draws of the
``rp_combdiv`` noise source (``kind: diverse_comb``) and two real reference
windows from the ``rp_all`` training sources.

Each draw is ``DiverseCombNoisePool.render`` on the policy's own noise spec,
with two changes for listening: 6 s windows instead of 2 s, and
``flight_reuse: 1`` so every draw is a fresh drone and flight (training cuts
32 windows from each). Writes ``<slug>.png``, ``<slug>.wav`` (mic 0, every clip
at the same RMS) and ``index.json`` next to this file.

    PYTHONPATH=src .venv/bin/python docs/explainers/diverse-comb/build.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import yaml

from data_processing.frames import get_meta
from data_processing.mixing import load_noise_source_frames
from data_processing.rotor_spectral_model import DiverseCombNoisePool, DiverseRotorProfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
POLICY = ROOT / "conf/online_mix/rp_combdiv.yaml"
SR = 16000
WINDOW_S = 6.0
N_DRAWS = 10
SEED = 20261022
MIC = 0
TARGET_RMS = 0.05
#: Random windows searched for one take-off and one landing example.
MAX_RAMP_TRIES = 400
N_FFT, HOP = 2048, 256
ZOOM_HZ = 1500.0
ROTOR_COLOURS = ("tab:blue", "tab:orange", "tab:green", "tab:red")
#: Real cruise windows from the rp_all sources: (recording, offset into the
#: recording in s, label).
REAL = (
    ("free-flight_nosource_room2", 30.0, "DREGON, free flight, room 2"),
    ("FLY125", 60.0, "MD2, FLY125"),
)


def level(x: np.ndarray) -> np.ndarray:
    return (x / (np.sqrt(np.mean(x**2)) + 1e-12) * TARGET_RMS).astype(np.float32)


def spectrogram_db(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    win = np.hanning(N_FFT + 1)[:N_FFT]
    starts = np.arange(0, x.size - N_FFT + 1, HOP)
    frames = np.stack([x[s : s + N_FFT] * win for s in starts])
    power = np.abs(np.fft.rfft(frames, axis=-1)) ** 2 / np.sum(win**2)
    t = (starts + N_FFT / 2) / SR
    f = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    return t, f, 10.0 * np.log10(power.T + 1e-20)


def line_levels(
    prof: DiverseRotorProfile, gain: float, median_rps: float, floor_std: float, nfft: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(k, dB over the floor in one nfft-point bin, kept)`` of one rotor's
    in-band lines at one microphone, as :func:`render_diverse_comb` counts
    them: 0 dB is the burial threshold."""
    ref = max(median_rps, 0.25 * 80.0)
    k = np.arange(1, prof.level_db.size + 1)
    band = k * ref < SR / 2
    speed = 2.5 * 20.0 * np.log10(ref / 80.0)
    in_bin = 10.0 * np.log10(nfft / 3.0)
    db = prof.level_db + 20.0 * np.log10(gain) + speed - 3.0103 + in_bin
    db = db - 20.0 * np.log10(floor_std)
    return k[band], db[band], prof.kept[band]


def plot_clip(
    x: np.ndarray,
    rps: np.ndarray,
    title: str,
    path: Path,
    profile: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None,
) -> None:
    t, f, s = spectrogram_db(x)
    top = float(np.percentile(s, 99.7))
    fig = plt.figure(figsize=(15, 4.6))
    cols = [2.4, 1.2, 1.7] if profile is not None else [2.4, 1.2]
    gs = fig.add_gridspec(2, len(cols), height_ratios=[3.2, 1.0], width_ratios=cols)
    ax_full = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_rps = fig.add_subplot(gs[1, 0:2])
    for ax, fmax in ((ax_full, SR / 2), (ax_zoom, ZOOM_HZ)):
        keep = f <= fmax
        ax.pcolormesh(
            t, f[keep] / 1000, s[keep], vmin=top - 70, vmax=top, cmap="magma", shading="auto"
        )
        ax.set_ylabel("kHz")
        ax.set_xticklabels([])
    ax_full.set_title(title, fontsize=10, loc="left")
    ax_zoom.set_title(f"0-{ZOOM_HZ / 1000:g} kHz", fontsize=9)
    tt = np.arange(rps.shape[1]) / SR
    for r in range(rps.shape[0]):
        ax_rps.plot(tt, rps[r], color=ROTOR_COLOURS[r], lw=1.0)
    ax_rps.set_xlim(0, tt[-1])
    ax_rps.set_ylabel("rev/s")
    if profile is not None:
        ax_p = fig.add_subplot(gs[:, 2])
        for r, (k, db, kept) in enumerate(profile):
            even = k % 2 == 0
            c = ROTOR_COLOURS[r]
            ax_p.scatter(k[kept & even], db[kept & even], s=10, color=c, label=f"rotor {r + 1}")
            ax_p.scatter(k[kept & ~even], db[kept & ~even], s=10, facecolors="none", edgecolors=c)
            ax_p.scatter(k[~kept], db[~kept], s=8, marker="x", color="0.75", lw=0.7)
        ax_p.axhline(0.0, color="black", ls="--", lw=1.0)
        ax_p.set_xscale("log")
        ax_p.set_xlabel("harmonic order k")
        ax_p.set_ylabel(f"line over floor, {N_FFT}-point spectrogram (dB)")
        ax_p.set_title("mic 0: filled even, hollow odd, x dropped", fontsize=9)
        ax_p.legend(fontsize=7, loc="upper right")
        ax_p.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_profiles(
    profiles: list[DiverseRotorProfile],
    levels: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    median: np.ndarray,
    path: Path,
) -> None:
    """One panel per rotor: every in-band line as a vertical stem (dark even,
    light odd; dropped lines as grey crosses at their drawn level), the two
    power laws the lines were drawn around, and the burial threshold."""
    fig, axes = plt.subplots(1, len(profiles), figsize=(15, 3.1), sharey=True)
    lo = min(float(db.min()) for _, db, _ in levels) - 3.0
    hi = max(float(db.max()) for _, db, _ in levels) + 3.0
    for r, (ax, prof, (k, db, kept)) in enumerate(zip(axes, profiles, levels, strict=True)):
        c = ROTOR_COLOURS[r]
        even = k % 2 == 0
        offset = float(db[0] - prof.level_db[k[0] - 1])  # profile dB -> dB over the floor
        ax.vlines(k[kept & even], lo, db[kept & even], color=c, lw=1.0)
        ax.vlines(k[kept & ~even], lo, db[kept & ~even], color=c, lw=1.0, alpha=0.35)
        ax.scatter(k[~kept], db[~kept], marker="x", s=10, color="0.6", lw=0.8, zorder=3)
        kk = np.linspace(1.0, float(k[-1]), 400)
        lk = np.log10(kk)
        ax.plot(kk, prof.gain_db - prof.even_slope_db * lk + offset, color="black", lw=0.9)
        odd_law = prof.gain_db - prof.odd_penalty_db - prof.odd_slope_db * lk + offset
        ax.plot(kk, odd_law, color="black", lw=0.9, ls=":")
        ax.axhline(0.0, color="black", ls="--", lw=0.8)
        ax.set_ylim(lo, hi)
        ax.set_xlim(0, k[-1] + 1)
        ax.set_xlabel("harmonic order k")
        ax.set_title(
            f"rotor {r + 1}, {median[r]:.0f} rev/s: even {prof.even_slope_db:.0f} dB/dec, "
            f"odd -{prof.odd_penalty_db:.0f} dB, drop {prof.drop_prob:.2f}",
            fontsize=8,
        )
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(f"over floor, {N_FFT}-pt (dB)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def emit(
    pool: DiverseCombNoisePool,
    audio: np.ndarray,
    rps: np.ndarray,
    profiles: list[DiverseRotorProfile],
    slug: str,
    label: str,
    kind: str,
) -> dict:
    """Write one rendered window's wav and figure; return its index entry."""
    info = pool.last_render
    assert info is not None
    median = np.median(rps, axis=1)
    prof_levels = [
        line_levels(
            p,
            float(info["mic_gains"][MIC, r]),
            float(median[r]),
            float(info["floor_std"][MIC]),
            pool.floor_ref_nfft,
        )
        for r, p in enumerate(profiles)
    ]
    x = level(audio[MIC])
    sf.write(HERE / f"{slug}.wav", x, SR)
    title = (
        f"{label}: median rotor speeds {', '.join(f'{v:.0f}' for v in median)} rev/s, "
        f"floor buries {info['subsumed']:.0%} of the most buried rotor's lines"
    )
    plot_clip(x, rps, title, HERE / f"{slug}.png", prof_levels)
    plot_profiles(profiles, prof_levels, median, HERE / f"{slug}_profiles.png")
    print("built", slug, flush=True)
    return {
        "slug": slug,
        "kind": kind,
        "label": label,
        "median_rps": [float(v) for v in median],
        "subsumed": float(info["subsumed"]),
        "buried_mic0": [float(v) for v in info["buried"][MIC]],
        "rotors": [
            {
                "even_slope_db": p.even_slope_db,
                "odd_penalty_db": p.odd_penalty_db,
                "odd_slope_db": p.odd_slope_db,
                "scatter_db": p.scatter_db,
                "drop_prob": p.drop_prob,
                "gain_db": p.gain_db,
            }
            for p in profiles
        ],
    }


def transition(rps: np.ndarray) -> str | None:
    """``take-off`` / ``landing`` when the window's rotor mean moves between
    idle and flight across it, else ``None``."""
    m = rps.mean(axis=0)
    third = m.size // 3
    first, last = float(np.median(m[:third])), float(np.median(m[-third:]))
    if last > 30.0 and 5.0 < first < 0.6 * last:
        return "take-off"
    if first > 30.0 and 5.0 < last < 0.6 * first:
        return "landing"
    return None


def main() -> None:
    spec = copy.deepcopy(yaml.safe_load(POLICY.read_text())["sources"]["noise"][0])
    spec["rps"]["flight_reuse"] = 1
    pool = DiverseCombNoisePool.from_config(spec, duration_s=WINDOW_S, sample_rate=SR)
    rng = np.random.default_rng(SEED)
    index: list[dict] = []
    for i in range(N_DRAWS):
        audio, rps, profiles = pool.render(rng, WINDOW_S)
        index.append(emit(pool, audio, rps, profiles, f"draw_{i:02d}", f"draw {i + 1}", "draw"))

    # The first take-off and the first landing among further random windows.
    rng = np.random.default_rng(SEED + 1)
    wanted = {"take-off", "landing"}
    for _ in range(MAX_RAMP_TRIES):
        audio, rps, profiles = pool.render(rng, WINDOW_S)
        found = transition(rps)
        if found in wanted:
            wanted.discard(found)
            slug = f"ramp_{found.replace('-', '')}"
            index.append(emit(pool, audio, rps, profiles, slug, found, "ramp"))
        if not wanted:
            break

    sources = yaml.safe_load((ROOT / "conf/online_mix/rp_all.yaml").read_text())["sources"]
    real = {
        str(get_meta(fr, "recording_id")): fr
        for fr in load_noise_source_frames(
            [s for s in sources["noise"] if s["kind"] == "frames"], sample_rate=SR
        )
    }
    for j, (rid, offset, label) in enumerate(REAL):
        fr = real[rid]
        a, r = fr["audio"], fr["rps"]
        start = int(offset * SR)
        n = int(WINDOW_S * SR)
        x = level(np.asarray(a.data)[MIC, start : start + n].astype(np.float64))
        t_audio = float(a.t_start) + (start + np.arange(n)) / SR
        rps = np.stack(
            [np.interp(t_audio, np.asarray(r.timestamps), row) for row in np.asarray(r.data)]
        )
        slug = f"real_{j}"
        sf.write(HERE / f"{slug}.wav", x, SR)
        median = np.median(rps, axis=1)
        title = (
            f"{label}, {offset:.0f}-{offset + WINDOW_S:.0f} s: median rotor speeds "
            + ", ".join(f"{v:.0f}" for v in median)
            + " rev/s"
        )
        plot_clip(x, rps, title, HERE / f"{slug}.png", None)
        index.append(
            {"slug": slug, "kind": "real", "label": label, "median_rps": [float(v) for v in median]}
        )
        print("built", slug, flush=True)
    (HERE / "index.json").write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()
