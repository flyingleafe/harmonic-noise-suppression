"""Are the v2 lines too THIN? Spectrograms, line widths, and widened variants.

The bench fits identified ``gamma_rk`` on stationary full-FFT bench recordings
and the flight fit froze them: under 1 Hz at every ``k <= 12``. The legacy
model's own DREGON cruise export renders its comb through a pedestal of
``13.07 + 0.211 k`` Hz half-width. This runner puts the two side by side:

* (A) spectrograms of mic 0 at 2048/512 and 8192/1024, 0-2 kHz, with the four
  label carriers' harmonics ``k = 1..8`` overlaid, for the real clip, the
  legacy render, the v2 R3 candidate as fitted and the same 21 dB up; and the
  -3 dB width of the order-tracked line at ``k = 1, 2, 4`` on the 8192 grid;
* (B) widened variants: ``gamma_rk`` scaled x3, x10, x30 and ``gamma_rk`` set
  to the legacy width law, each at the fitted comb level and 12 dB up, scored
  with the frozen HPPNet.

    python scripts/noise_v2_widen_dregon.py --probe --figures --out DIR
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from experiments.noise_model import gates as GT
from experiments.noise_model import render as RD
from experiments.stochastic_fit import revised_eval as RE

SCHEMA = "noise-v2-dregon-widen/1"
OUT_DEFAULT = Path("results/noise_v2/rounds/round3/dregon_humps")
FIT_DEFAULT = Path("results/noise_v2/rounds/round3/fits/dregon_room2_floor__flight_floor_lowk.json")
RECORDINGS = (
    "free-flight_nosource_room2",
    "hovering_nosource_room2",
    "updown_nosource_room2",
)
SEED = 2001
WIDE_N, WIDE_HOP = 8192, 1024
SPEC_N, SPEC_HOP = 2048, 512
WIDTH_ORDERS = (1, 2, 4)
LEGACY_G0, LEGACY_SLOPE = 13.068, 0.211
SCALES = (3.0, 10.0, 30.0)
SHIFTS = (0.0, 12.0)
MUTE_DB = 200.0


def die(m: str) -> NoReturn:
    raise SystemExit(f"error: {m}")


def _module(name: str) -> Any:
    path = Path("scripts") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        die(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def mutate(
    fit: dict[str, Any], *, gamma_scale: float = 1.0, gamma_law: bool = False, shift_db: float = 0.0
) -> dict[str, Any]:
    out = copy.deepcopy(fit)
    p = out["params"]
    g = np.asarray(p["gamma_hz"], dtype=np.float64)
    if gamma_law:
        k = np.arange(1, g.shape[1] + 1, dtype=np.float64)
        g = np.broadcast_to(LEGACY_G0 + LEGACY_SLOPE * k, g.shape).copy()
    elif gamma_scale != 1.0:
        g = g * float(gamma_scale)
    p["gamma_hz"] = g.tolist()
    if shift_db:
        p["profile"]["profile_db"] = (
            np.asarray(p["profile"]["profile_db"], dtype=np.float64) + float(shift_db)
        ).tolist()
    return out


def stft_power(x: np.ndarray, *, n: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    w = np.hanning(n)
    starts = np.arange(0, x.size - n + 1, hop)
    power = np.stack([np.abs(np.fft.rfft(x[s : s + n] * w)) ** 2 for s in starts])
    return power / float((w**2).sum()), starts + n // 2


def line_width_db3(audio: np.ndarray, f0_tracks: np.ndarray, k: int, *, sr: int) -> dict[str, Any]:
    """-3 dB (half-power) width of the order-tracked line on the 8192 grid.

    Each frame's spectrum is read on that frame's own label carrier for each
    rotor and averaged (mics and rotors), so the drift does not smear the
    average; the width inside one frame (512 ms) and the four-rotor spread are
    still in the number and are the same for every arm.
    """
    freqs = np.fft.rfftfreq(WIDE_N, d=1.0 / float(sr))
    f0 = np.asarray(f0_tracks, dtype=np.float64)
    fbar = float(f0.mean())
    grid = np.arange(-0.75 * fbar, 0.75 * fbar + 1e-9, 0.5)
    acc = np.zeros(grid.size)
    n_used = 0
    for m in range(int(audio.shape[0])):
        power, centres = stft_power(audio[m], n=WIDE_N, hop=WIDE_HOP)
        idx = np.clip(centres.astype(np.int64), 0, int(f0.shape[1]) - 1)
        for r in range(int(f0.shape[0])):
            car = f0[r][idx] * float(k)
            for j, c in enumerate(car):
                acc += np.interp(c + grid, freqs, power[j])
                n_used += 1
    prof = acc / max(n_used, 1)
    side = (np.abs(grid) >= 0.45 * fbar) & (np.abs(grid) <= 0.7 * fbar)
    base = float(np.median(prof[side])) if side.any() else float(np.median(prof))
    ex = prof - base
    i0 = int(np.argmin(np.abs(grid)))
    peak = float(ex[max(i0 - 2, 0) : i0 + 3].max())
    if peak <= 0:
        return dict(
            k=int(k),
            width_hz=None,
            peak_over_base_db=float("nan"),
            resolution_hz=float(sr) / WIDE_N,
        )
    half = peak / 2.0

    def cross(step: int) -> float:
        j = i0
        while 0 <= j + step < ex.size and abs(float(grid[j + step])) <= 0.5 * fbar:
            if ex[j + step] <= half:
                y0, y1 = float(ex[j]), float(ex[j + step])
                if y1 == y0:
                    return float(grid[j + step])
                return float(grid[j]) + (half - y0) * (float(grid[j + step]) - float(grid[j])) / (
                    y1 - y0
                )
            j += step
        return float("nan")

    lo, hi = cross(-1), cross(1)
    return dict(
        k=int(k),
        width_hz=(float(hi - lo) if np.isfinite(lo) and np.isfinite(hi) else None),
        peak_over_base_db=float(10.0 * np.log10(max(peak + base, 1e-300) / max(base, 1e-300))),
        resolution_hz=float(sr) / WIDE_N,
        four_rotor_span_hz=float(k * (f0.mean(axis=1).max() - f0.mean(axis=1).min())),
    )


@dataclass(frozen=True)
class Arm:
    name: str
    label: str
    kind: str = "v2"
    gamma_scale: float = 1.0
    gamma_law: bool = False
    shift_db: float = 0.0


def arms() -> tuple[Arm, ...]:
    out = [
        Arm("real", "the real DREGON room-2 clip", kind="real"),
        Arm("legacy", "legacy stage-2 baseline, identity-matched", kind="legacy"),
        Arm("v2", "the round-3 v2 candidate as fitted"),
        Arm("v2_plus21db", "v2 with the comb 21 dB up (the level optimum)", shift_db=21.0),
    ]
    for sh in SHIFTS:
        tag = "" if sh == 0.0 else f"_plus{int(sh)}db"
        for s in SCALES:
            out.append(
                Arm(
                    f"v2_gx{int(s)}{tag}",
                    f"gamma_rk x{s:g}" + ("" if sh == 0 else f", comb {sh:g} dB up"),
                    gamma_scale=s,
                    shift_db=sh,
                )
            )
        out.append(
            Arm(
                f"v2_glegacy{tag}",
                f"gamma_rk = {LEGACY_G0:g} + {LEGACY_SLOPE:g}k Hz (the legacy width law)"
                + ("" if sh == 0 else f", comb {sh:g} dB up"),
                gamma_law=True,
                shift_db=sh,
            )
        )
    return tuple(out)


def _probe(experiment: str | None, ckpt: str) -> Any:
    """The frozen HPPNet probe, or any other checkpoint as a SECOND tracker."""
    rs = _module("noise_v2_round_score")
    if experiment is None:
        return rs.Probe.load()
    ev = _module("stochastic_fit_revised_eval")
    tracker = ev.Tracker(dict(experiment=str(experiment), ckpt=str(ckpt)))
    return type("P", (), dict(record=tracker.record, tracker=tracker))()


def run(
    *,
    fit_path: Path,
    out: Path,
    probe: bool,
    recordings: tuple[str, ...],
    only: tuple[str, ...] = (),
    probe_experiment: str | None = None,
    probe_ckpt: str = "best",
) -> dict[str, Any]:
    rs = _module("noise_v2_round_score")
    fit = json.loads(Path(fit_path).read_text())
    probe_obj = _probe(probe_experiment, probe_ckpt) if probe else None
    legacy = rs.legacy_arm("dregon")
    specs = tuple(a for a in arms() if not only or a.name in only)
    payload: dict[str, Any] = dict(
        schema=SCHEMA,
        git=git_rev(),
        fit=dict(path=str(fit_path), mode=fit.get("mode")),
        protocol=dict(
            seed=SEED,
            mic=0,
            wide=dict(n_fft=WIDE_N, hop=WIDE_HOP, resolution_hz=16000.0 / WIDE_N),
            spec=dict(n_fft=SPEC_N, hop=SPEC_HOP),
            width_orders=list(WIDTH_ORDERS),
            legacy_law=dict(gamma0_hz=LEGACY_G0, slope_hz_per_order=LEGACY_SLOPE),
            scales=list(SCALES),
            shifts_db=list(SHIFTS),
            scorer=(probe_obj.record if probe_obj is not None else None),
            arms=[dict(name=a.name, label=a.label) for a in specs],
            fitted_gamma_hz={
                str(k): float(
                    np.asarray(fit["params"]["gamma_hz"], dtype=np.float64)[:, k - 1].mean()
                )
                for k in (1, 2, 4, 8)
            },
        ),
        supports={},
    )
    figures: dict[str, Any] = {}
    for recording in recordings:
        found = [s for s in GT.DREGON_CRUISE_SUPPORTS if s.recording == recording]
        if not found:
            die(f"no frozen DREGON cruise support carries recording {recording!r}")
        support = found[0]
        clip = RE.load_window(
            support.window,
            dataset=GT.DATASET["dregon"],
            version=None,
            channels=None,
            rps_key=GT.RAW_RPS_KEY["dregon"],
        )
        sr = int(clip.sr)
        real = np.asarray(clip.audio, dtype=np.float64)
        reference = np.atleast_2d(np.asarray(clip.rps, dtype=np.float64))
        rsupport = RE.regime_support(
            support.window,
            reference,
            regime=support.regime,
            min_rps=support.min_rps,
            max_rps=support.max_rps,
            sr=sr,
        )
        larm = rs._arm_for_recording(legacy, support)
        row: dict[str, Any] = dict(support=support.as_dict(), arms={})
        waves: dict[str, np.ndarray] = {}
        for spec in specs:
            if spec.kind == "real":
                audio = real
            elif spec.kind == "legacy":
                audio = larm.render(reference, regime=support.regime, n_mics=8, seed=SEED)
            else:
                audio = RD.render_noise(
                    mutate(
                        fit,
                        gamma_scale=spec.gamma_scale,
                        gamma_law=spec.gamma_law,
                        shift_db=spec.shift_db,
                    ),
                    reference,
                    n_mics=8,
                    seed=SEED,
                )
            audio = np.asarray(audio, dtype=np.float64)[:8]
            entry: dict[str, Any] = dict(
                label=spec.label,
                kind=spec.kind,
                widths=[line_width_db3(audio, reference, k, sr=sr) for k in WIDTH_ORDERS],
            )
            if probe_obj is not None:
                pit = probe_obj.tracker.pit(
                    audio,
                    reference,
                    mics=list(range(8)),
                    expected_samples=int(real.shape[-1]),
                    support=rsupport,
                )
                entry["pit_mae"] = float(pit["mae"])
            row["arms"][spec.name] = entry
            if spec.name in ("real", "legacy", "v2", "v2_plus21db"):
                waves[spec.name] = audio[0].copy()
            w0 = entry["widths"][0]["width_hz"]
            print(
                f"[{recording}] {spec.name}: k1 width "
                + ("—" if w0 is None else f"{w0:.1f} Hz")
                + f" pit={entry.get('pit_mae')}",
                flush=True,
            )
        payload["supports"][support.key] = row
        figures[recording] = dict(waves=waves, rps=reference, sr=sr)
    Path(out).mkdir(parents=True, exist_ok=True)
    return payload | {"_figures": figures}


def write_figures(figures: dict[str, Any], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[str] = []
    names = ("real", "legacy", "v2", "v2_plus21db")
    for recording, blob in figures.items():
        rps = blob["rps"]
        sr = int(blob["sr"])
        for n_fft, hop, tag in ((SPEC_N, SPEC_HOP, "2048"), (WIDE_N, WIDE_HOP, "8192")):
            fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), layout="constrained", sharey=True)
            for ax, name in zip(axes, names, strict=True):
                x = blob["waves"].get(name)
                if x is None:
                    continue
                power, centres = stft_power(x, n=n_fft, hop=hop)
                freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
                sel = freqs <= 2000.0
                t = centres / float(sr)
                db = 10.0 * np.log10(np.maximum(power[:, sel], 1e-20)).T
                im = ax.imshow(
                    db,
                    origin="lower",
                    aspect="auto",
                    extent=(float(t[0]), float(t[-1]), 0.0, 2000.0),
                    cmap="magma",
                    vmin=float(np.percentile(db, 20)),
                    vmax=float(db.max()),
                )
                idx = np.clip((t * sr).astype(np.int64), 0, int(rps.shape[1]) - 1)
                for k in range(1, 9):
                    for r in range(int(rps.shape[0])):
                        ax.plot(t, rps[r, idx] * k, color="cyan", lw=0.4, alpha=0.55)
                ax.set_title(f"{name} ({tag}/{hop})")
                ax.set_xlabel("s")
            axes[0].set_ylabel("Hz")
            fig.colorbar(im, ax=axes[-1], label="dB")
            path = out / f"spec_{recording.split('_')[0]}_{tag}.png"
            fig.savefig(path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            written.append(str(path))
    return written


def findings(payload: dict[str, Any], figures: list[str], *, job: str | None) -> str:
    rows = list(payload["supports"].values())
    recs = [str(r["support"]["recording"]).split("_")[0] for r in rows]
    names = [a["name"] for a in payload["protocol"]["arms"]]
    o = ["# DREGON round 3: spectrograms and widened lines", ""]
    o.append(
        f"Mic 0, seed {payload['protocol']['seed']}; spectrograms at "
        f"{SPEC_N}/{SPEC_HOP} and {WIDE_N}/{WIDE_HOP} "
        f"({payload['protocol']['wide']['resolution_hz']:.2f} Hz per bin), 0-2 kHz, label "
        "harmonics k=1..8 overlaid. The fitted `gamma_rk` (mean over rotors) is "
        + ", ".join(f"k={k}: {v:.4f} Hz" for k, v in payload["protocol"]["fitted_gamma_hz"].items())
        + " against the legacy law "
        f"{LEGACY_G0:g} + {LEGACY_SLOPE:g}k Hz."
    )
    o.append("")
    o.append("## -3 dB width of the order-tracked line (Hz, 8192-point, 1.95 Hz per bin)")
    o.append("")
    o.append(
        "Each frame read on its own label carrier, averaged over four rotors and eight "
        "microphones; the four-rotor spread (k x ~7.8 rev/s) and the 512 ms frame's own "
        "drift are inside every number equally."
    )
    o.append("")
    o.append(
        "| window | arm | "
        + " | ".join(f"k={k}" for k in WIDTH_ORDERS)
        + " | peak over local base, k=1 |"
    )
    o.append("|---|---|" + "---:|" * (len(WIDTH_ORDERS) + 1))
    for r, rec in zip(rows, recs, strict=True):
        for n in names:
            e = r["arms"][n]
            cells = []
            for w in e["widths"]:
                cells.append("—" if w["width_hz"] is None else f"{w['width_hz']:.1f}")
            o.append(
                f"| `{rec}` | `{n}` | "
                + " | ".join(cells)
                + f" | {e['widths'][0]['peak_over_base_db']:+.2f} dB |"
            )
    o.append("")
    if any("pit_mae" in r["arms"][n] for r in rows for n in names):
        o.append("## HPPNet PIT MAE (rev/s)")
        o.append("")
        o.append("| arm | " + " | ".join(recs) + " | mean |")
        o.append("|---|" + "---:|" * (len(recs) + 1))
        for n in names:
            v = [r["arms"][n].get("pit_mae") for r in rows]
            ok = [x for x in v if x is not None]
            o.append(
                f"| `{n}` | "
                + " | ".join("—" if x is None else f"{x:.3f}" for x in v)
                + f" | {(np.mean(ok) if ok else float('nan')):.3f} |"
            )
        o.append("")
    o.append("| arm | what it is |")
    o.append("|---|---|")
    for a in payload["protocol"]["arms"]:
        o.append(f"| `{a['name']}` | {a['label']} |")
    o.append("")
    if figures:
        o.append("Figures: " + ", ".join(f"`{p}`" for p in figures) + ".")
        o.append("")
    o.append(f"Record `{payload['git']}`" + (f", job `{job}`." if job else "."))
    o.append("")
    return "\n".join(o)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", type=Path, default=FIT_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--job", default=None)
    ap.add_argument("--recordings", default=",".join(RECORDINGS))
    ap.add_argument("--arms", default="", help="comma-separated arm subset")
    ap.add_argument("--probe-experiment", default=None, help="a SECOND tracker's experiment")
    ap.add_argument("--probe-ckpt", default="best")
    ap.add_argument("--stem", default="widen", help="output file stem")
    args = ap.parse_args(argv)
    payload = run(
        fit_path=args.fit,
        out=args.out,
        probe=bool(args.probe),
        recordings=tuple(r for r in str(args.recordings).split(",") if r),
        only=tuple(a for a in str(args.arms).split(",") if a),
        probe_experiment=args.probe_experiment,
        probe_ckpt=str(args.probe_ckpt),
    )
    figs = payload.pop("_figures")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = write_figures(figs, out) if args.figures else []
    (out / f"{args.stem}.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    (out / f"{args.stem}.md").write_text(findings(payload, written, job=args.job))
    print(f"wrote {out / (args.stem + '.json')}")
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
