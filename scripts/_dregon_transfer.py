"""Does a rig fitted on the DREGON BENCH predict DREGON IN FLIGHT?

Michael's fit saw flight recordings. This does not: the parameters come from the
four clamped single-motor bench runs, and the only thing taken from the flight
is its rotor telemetry (the carriers) — no level, no floor, no spectrum.

Three arms, so the transfer gap can be attributed:

* **bench, blind** — every number from the bench, including its room floor.
* **bench comb + flight floor** — the comb from the bench, with the broadband
  floor's LEVEL matched to the real clip in a line-free band (one number).
  Isolates "is the comb right" from "is the room right".
* **flight-fitted** — the same model fitted to the flight recording, the
  reference a transfer can be measured against (optional, ``--flight-fit``).

Each arm is rendered at the real clip's own rotor trajectories, then scored by
the frozen ``hppnet_l2_r2_s0`` on all eight channels, and compared to the real
clip through the same pipeline.

    python scripts/_dregon_transfer.py --bench-fit results/S1/bayes_rig.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import zoo
from data_processing import stochastic_rotor_noise as srn
from experiments.stochastic_fit import accept_stats as stats
from experiments.stochastic_fit import clips as C
from experiments.stochastic_fit import stage2 as S2
from experiments.stochastic_fit.data import Clip
from metrics.salience_layers import LayerPeakRPSMetric

OUT = Path("docs/explainers/dregon-transfer")
RESULTS = Path("results/S1/transfer")
MODEL = "hppnet_l2_r2_s0"
RECORDING = "free-flight_nosource_room2"
#: line-free band used to match the floor LEVEL of the second arm
FLOOR_BAND = (5000.0, 7000.0)


def _probe_helpers() -> Any:
    path = Path(__file__).resolve().parent / "_synthetic_probe.py"
    spec = importlib.util.spec_from_file_location("_synthetic_probe", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rig_params_from_bench(
    bench: dict[str, Any], rates: np.ndarray, *, n_mics: int, sample_rate: int
) -> Any:
    """Renderer parameters for a four-rotor flight, built from BENCH cells.

    One bench cell per rotor, chosen as the cell whose motor matches the rotor
    index and whose setpoint is closest to that rotor's flight rate. The cell's
    own speed law then carries it the rest of the way, so no level is taken from
    the flight.

    The bench is single-channel, so it cannot supply a per-microphone pattern:
    the microphone gains are left flat. That is a stated limitation of the
    transfer, not a modelling choice.

    A word on what "per rotor" can mean here. A rig fit over bench clips has
    ``R = 1`` (one rotor per clip), so its tied profile is ONE shape plus a
    scalar level per clip: passing a single rig file gives four rotors the same
    shape at four different levels. Passing four SEPARATE single-motor fits
    gives four genuinely different shapes, which is what the measurement asks
    for - Motor 2 sits 5 to 13 dB below Motor 1 at 1 kHz, and that difference is
    frequency dependent, not a gain.
    """
    cells = bench["clips"]
    profiles, gamma0, slope, k_half = [], [], [], []
    picked = []
    for r, rate in enumerate(np.atleast_1d(rates)):
        motor = r + 1
        cand = {c: e for c, e in cells.items() if c.startswith(f"Motor{motor}_")}
        if not cand:  # a single-motor fit: every rotor inherits the same motor
            cand = cells
        best = min(cand, key=lambda c: abs(float(c.split("_")[1]) - float(rate)))
        picked.append(best)
        p = cand[best]["params"]
        prof = np.atleast_2d(np.asarray(p["profile_db"], dtype=np.float64))
        h = np.asarray(p.get("h_db", 0.0), dtype=np.float64)
        if h.ndim == 3:
            prof = prof + h.mean(axis=-1)[: prof.shape[0]]
        profiles.append(prof[0])
        gamma0.append(float(np.atleast_1d(p["gamma0"])[0]))
        slope.append(float(np.atleast_1d(p["gamma_slope"])[0]))
        k_half.append(float(p.get("coherence_k_half", 0.0)))
    ref = cells[picked[0]]["params"]
    k_max = max(2, int(np.floor((sample_rate / 2) / max(float(np.min(rates)), 1.0))))
    k_use = min(k_max, min(len(p) for p in profiles))
    return (
        srn.StochasticParams(
            sample_rate=int(sample_rate),
            n_rotors=len(profiles),
            n_harmonics=k_use,
            profile_db=np.stack([p[:k_use] for p in profiles]).copy(),
            gamma0=np.asarray(gamma0),
            gamma_slope=np.asarray(slope),
            floor_ctrl_hz=np.asarray(ref["floor_ctrl_hz"], dtype=np.float64),
            floor_ctrl_db=np.asarray(ref["floor_shape_db"], dtype=np.float64),
            floor_tilt_db_oct=float(ref.get("floor_tilt_db_oct", 0.0)),
            harm_mean_db=0.0,
            floor_mean_db=float(ref.get("floor_mean_db", 0.0)),
            harm_gp_std_db=0.0,
            harm_gp_tau_s=1.0,
            harm_coherence=0.0,
            floor_gp_std_db=0.0,
            floor_gp_tau_s=1.0,
            floor_tilt_gp_std=0.0,
            floor_tilt_gp_tau_s=1.0,
            line_bin_integrate=True,
            floor_static_rel=float(ref.get("floor_static_rel", 0.0)),
            amp_rps_exponent=float(ref.get("amp_exp", 0.0)),
            amp_rps_exponent_floor=float(ref.get("floor_exp", ref.get("amp_exp", 0.0))),
            amp_rps_ref=80.0,
            shaft_jitter_rps=0.0,
            shaft_jitter_tau_s=2.0,
            phase_diffusion_hz_per_order=0.0,
            shaft_offset_rps=0.0,
            umod_std_db=0.0,
            umod_tau_s=1.0,
            umod_corner_hz=200.0,
            mic_gain_all_db=0.0,
            mic_floor_std_db=0.0,
            gamma_min_bins=0.01,
            coherence_k_half=float(np.mean(k_half)),
        ),
        picked,
    )


def render(params: Any, rps16: np.ndarray, *, n_mics: int, seed: int) -> np.ndarray:
    """``(M, T)`` 16 kHz audio, rendered natively and decimated."""
    n_native = int(round(rps16.shape[-1] / S2.SR * C.NATIVE_SR))
    t_src = np.linspace(0.0, 1.0, rps16.shape[-1])
    t_dst = np.linspace(0.0, 1.0, n_native)
    rps_native = np.stack([np.interp(t_dst, t_src, r) for r in rps16])
    audio, _ = srn.synthesize(
        params,
        rps_native,
        rng=np.random.default_rng(seed),
        n_mics=n_mics,
        line_mode="fm",
        n_fft=1 << 16,
    )
    clip = Clip("t", "synthetic", np.asarray(audio, np.float32), rps_native, C.NATIVE_SR)
    return np.asarray(C.decimate(clip, S2.SR).audio, dtype=np.float64)


def band_level_db(x: np.ndarray, lo: float, hi: float, sr: int = S2.SR) -> float:
    n = 1 << 14
    w = np.hanning(n)
    fr = np.stack([x[i : i + n] * w for i in range(0, max(x.size - n, 1), n)])
    P = (np.abs(np.fft.rfft(fr, axis=-1)) ** 2).mean(0)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    sel = (f >= lo) & (f <= hi)
    return float(10.0 * np.log10(P[sel].mean() + 1e-30))


def spec_panel(ax, x: np.ndarray, title: str, sr: int = S2.SR) -> None:
    n, hop = 2048, 512
    w = np.hanning(n + 1)[:n]
    fr = np.stack([x[s : s + n] * w for s in range(0, x.size - n, hop)])
    S = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=-1)).T + 1e-9)
    v = np.percentile(S, 99.5)
    ax.imshow(
        S,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, x.size / sr, 0, sr / 2000],
        vmin=v - 70,
        vmax=v,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bench-fit",
        type=Path,
        nargs="+",
        default=[Path("results/S1/bayes_rig.json")],
        help="one or more bench fits; several files give a real per-motor shape, "
        "one rig file gives ONE shared shape with per-cell levels",
    )
    ap.add_argument("--flight-fit", type=Path, default=None)
    ap.add_argument("--recording", default=RECORDING)
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--min-rps", type=float, default=60.0)
    ap.add_argument("--dataset", default=S2.DREGON_DATASET, help="frames dataset, NAME[@VERSION]")
    ap.add_argument("--version", default=None)
    ap.add_argument("--channels", default=None, help="'all' (default) or e.g. '0,3-5'")
    ap.add_argument("--rps-key", default=C.DEFAULT_RPS_KEY)
    ap.add_argument("--out", type=Path, default=RESULTS / "transfer.json")
    args = ap.parse_args()

    bench = {"clips": {}}
    for f in args.bench_fit:
        bench["clips"].update(json.loads(f.read_text())["clips"])
    print(f"bench source: {len(bench['clips'])} cells from {len(args.bench_fit)} file(s)")
    flight_fit = json.loads(args.flight_fit.read_text()) if args.flight_fit else None
    helpers = _probe_helpers()
    fm = zoo.load(MODEL, ckpt="best")
    metric = LayerPeakRPSMetric()
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    windows = S2.cruise_windows(
        args.recording,
        dataset=args.dataset,
        version=args.version,
        rps_key=args.rps_key,
        seconds=args.seconds,
        min_rps=args.min_rps,
        max_clips=args.clips,
        stride_s=args.seconds,
    )
    if not windows:
        raise SystemExit(f"{args.recording}: no window of {args.seconds}s above {args.min_rps}")

    rows: list[dict[str, Any]] = []
    for i, (start_s, dur) in enumerate(windows):
        real_clip = C.decimate(
            C.load_clip(
                args.dataset,
                args.recording,
                start_s,
                dur,
                version=args.version,
                channels=args.channels,
                rps_key=args.rps_key,
                clip_id=f"transfer_{args.recording}_{start_s:.2f}",
            ),
            S2.SR,
        )
        xr_all = np.asarray(real_clip.audio, dtype=np.float64)
        rps = np.asarray(real_clip.rps, dtype=np.float64)
        m = int(xr_all.shape[0])
        rates = rps.mean(axis=1)

        params, picked = rig_params_from_bench(bench, rates, n_mics=m, sample_rate=C.NATIVE_SR)
        arms: dict[str, np.ndarray] = {"bench, blind": render(params, rps, n_mics=m, seed=500 + i)}
        # second arm: the same comb with the floor level matched in a line-free band
        d = band_level_db(xr_all[0], *FLOOR_BAND) - band_level_db(
            arms["bench, blind"][0], *FLOOR_BAND
        )
        arms["bench comb + flight floor"] = render(
            params.with_(floor_mean_db=params.floor_mean_db + d), rps, n_mics=m, seed=500 + i
        )
        if flight_fit is not None:
            e = list(flight_fit["clips"].values())[i % len(flight_fit["clips"])]["params"]
            arms["flight-fitted"] = S2.render_from_export(e, rps, n_mics=m, seed=500 + i)

        n = min([xr_all.shape[-1]] + [a.shape[-1] for a in arms.values()])
        xr_all = xr_all[:, :n]
        arms = {k: v[:, :n] for k, v in arms.items()}
        rps_i = rps[:, :n]

        row: dict[str, Any] = dict(
            clip=i,
            start_s=start_s,
            rotors=np.round(rates, 1).tolist(),
            bench_cells=picked,
            floor_offset_db=round(float(d), 2),
            arms={},
        )
        for tag, audio in {"real": xr_all, **arms}.items():
            per_mic = []
            pred0 = truth0 = None
            for mic in range(m):
                pred, truth, mae = helpers.score(fm, metric, audio, rps_i, S2.SR, mic)
                per_mic.append(mae)
                if mic == 0:
                    pred0, truth0 = pred, truth
            assert pred0 is not None and truth0 is not None
            slug = tag.replace(", ", "_").replace(" ", "_").replace("+", "plus")
            fig_name = helpers.figure(
                f"dregon_{slug}", i, audio, rps_i, S2.SR, pred0, truth0, per_mic
            )
            ltas = stats.ltas_bands(audio[0])
            row["arms"][tag] = dict(
                mae_8mic=round(float(np.mean(per_mic)), 3),
                mae_mic0=round(float(per_mic[0]), 3),
                spread=round(float(np.max(per_mic) - np.min(per_mic)), 3),
                per_rotor=[round(float(v), 3) for v in np.abs(pred0 - truth0).mean(1)],
                ltas=np.round(ltas, 2).tolist(),
                figure=fig_name,
            )
            y = audio[0] / max(float(np.abs(audio[0]).max()), 1e-9) * 0.7
            sf.write(OUT / f"dregon_{i:02d}_{slug}.wav", y.astype(np.float32), S2.SR)

        base = np.asarray(row["arms"]["real"]["ltas"])
        for tag in arms:
            dev = np.abs(np.asarray(row["arms"][tag]["ltas"]) - base)
            row["arms"][tag]["ltas_mean_abs_db"] = round(float(dev.mean()), 2)
            row["arms"][tag]["ltas_max_abs_db"] = round(float(dev.max()), 2)

        names = ["real", *arms]
        fig, axes = plt.subplots(
            1,
            len(names) + 1,
            figsize=(5.2 * (len(names) + 1), 4.2),
            gridspec_kw={"width_ratios": [1] * len(names) + [1.15]},
        )
        for ax, tag in zip(axes[: len(names)], names, strict=True):
            src = xr_all if tag == "real" else arms[tag]
            spec_panel(ax, src[0], f"DREGON flight {i} — {tag}")
        axes[0].set_ylabel("kHz")
        centres = [(lo + hi) / 2000 for lo, hi in stats.BANDS]
        for tag, colour in zip(names, ["#111111", "#1f77b4", "#2ca02c", "#ff7f0e"], strict=False):
            axes[-1].plot(
                centres,
                row["arms"][tag]["ltas"],
                marker="o",
                ms=4,
                lw=1.4,
                color=colour,
                label=f"{tag} ({row['arms'][tag].get('mae_8mic')} rev/s)",
            )
        axes[-1].set_xscale("log")
        axes[-1].set_xlabel("kHz")
        axes[-1].set_ylabel("dB re 200-400 Hz")
        axes[-1].grid(alpha=0.3)
        axes[-1].legend(fontsize=7)
        axes[-1].set_title("LTAS, and the tracker's 8-mic MAE", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / f"dregon_{i:02d}.png", dpi=100)
        plt.close(fig)
        row["figure"] = f"dregon_{i:02d}.png"
        rows.append(row)
        print(
            f"clip {i} rotors {row['rotors']} cells {picked}: "
            + "  ".join(f"{t}={row['arms'][t]['mae_8mic']}" for t in ["real", *arms]),
            flush=True,
        )

    summary = {
        t: dict(
            median_mae_8mic=round(float(np.median([r["arms"][t]["mae_8mic"] for r in rows])), 3),
            median_spread=round(float(np.median([r["arms"][t]["spread"] for r in rows])), 3),
            median_ltas_dev_db=(
                None
                if t == "real"
                else round(float(np.median([r["arms"][t]["ltas_mean_abs_db"] for r in rows])), 2)
            ),
        )
        for t in rows[0]["arms"]
    }
    print("\n" + json.dumps(summary, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(summary=summary, clips=rows), indent=1))
    (OUT / "index.json").write_text(json.dumps(dict(summary=summary, clips=rows), indent=1))
    print(f"wrote {args.out} and {len(rows)} panels to {OUT}")


if __name__ == "__main__":
    main()
