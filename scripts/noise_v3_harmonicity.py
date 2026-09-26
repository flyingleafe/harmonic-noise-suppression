"""Is the legacy noise more TONAL than v2 / v3 on the real validation flights?

Dmitrii's demodulated prominence (:func:`experiments.noise_model.tonality.
demod_prominence`) on every noise family the SCv2 arms trained on, all rendered
on the SAME trajectories: the rotor tracks of the real validation clips
(``dload:DREGON-LM-V4-michaels-valid-full``, the telemetry labels stretched onto
the clip exactly as ``DregonLMFrameDataset`` stretches them), plus the
recordings of those clips themselves. A recording is read on its labels, which
carry their own error, so its number is a LOWER bound on its tonality.

    python scripts/noise_v3_harmonicity.py --subsets           # laptop, once
    python scripts/noise_v3_harmonicity.py --run --workers 32  # uni-cpu
    python scripts/noise_v3_harmonicity.py --plot              # laptop

``--subsets`` cuts :data:`SUBSET_INDICES` (16 entries) out of each of the six
banks into ``results/noise_v3/harmonicity/banks/`` (committed, so a job needs no
``data/`` tree). ``--run`` renders every source on every flying clip and writes
``results/noise_v3/harmonicity/prominence.json``. ``--plot`` reduces that into
the explainer's ``harm_summary.json`` and ``harm_prominence.png``. ``--smoke``
(with ``--run``) takes one source per family and one clip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/noise_v3/harmonicity"
SUBSET_DIR = OUT / "banks"
RESULT = OUT / "prominence.json"
DOCS = ROOT / "docs/explainers/noise-model-v3-latent-runaway"
SUMMARY = DOCS / "harm_summary.json"
FIGURE = DOCS / "harm_prominence.png"

VALID = "dload:DREGON-LM-V4-michaels-valid-full"
SR = 16000
#: The microphones every clip is read on (renders and recordings alike).
N_MICS = 4
#: A clip is a trajectory only if EVERY rotor stays at or above this speed for
#: the whole clip: ground, take-off and landing clips have no comb to read.
MIN_CLIP_RPS = 20.0
#: The clip recordings with a loudspeaker playing into the room. Their tracks
#: are real flights, but their audio holds a second, acoustic source.
LOUDSPEAKER = ("speech", "whitenoise")

#: Every bank is read at the same 16 entries, evenly over its file order (the
#: easy banks hold their DREGON half first, so this is 8 + 8).
SUBSET_INDICES = tuple(range(0, 2048, 128))
BANKS = {
    "legacy_easy": "data/rig_banks/rig_easy_n2048.json",
    "legacy_hard": "data/rig_banks/rig_hard_n2048.json",
    "v2_easy": "data/rig_banks/noise_v2_easy_n2048.json",
    "v2_hard": "data/rig_banks/noise_v2_hard_n2048.json",
    "v3_easy": "data/rig_banks/noise_v3r3_easy_n2048.json",
    "v3_hard": "data/rig_banks/noise_v3r3_hard_n2048.json",
}
#: ``(family, label, colour)`` in plotting order; ``real`` is the grey band.
FAMILIES = (
    ("legacy_fit", "legacy fits", "#222222"),
    ("legacy_easy", "legacy easy bank", "#1f77b4"),
    ("legacy_hard", "legacy hard bank", "#d62728"),
    ("v2_fit", "v2 fits", "#222222"),
    ("v2_easy", "v2 easy bank", "#1f77b4"),
    ("v2_hard", "v2 hard bank", "#d62728"),
    ("v3_fit", "v3 r3b fits", "#222222"),
    ("v3_easy", "v3r3 easy bank", "#1f77b4"),
    ("v3_hard", "v3r3 hard bank", "#d62728"),
)


def _subset_path(family: str) -> Path:
    return SUBSET_DIR / f"{family}_sub{len(SUBSET_INDICES)}.json"


# ---------------------------------------------------------------------------
# --subsets
# ---------------------------------------------------------------------------


def make_subsets() -> None:
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    for family, rel in BANKS.items():
        src = ROOT / rel
        blob = src.read_bytes()
        raw = json.loads(blob)
        sub = {k: v for k, v in raw.items() if k != "entries"}
        sub["entries"] = [raw["entries"][i] for i in SUBSET_INDICES]
        sub["subset"] = {
            "of": rel,
            "sha256": hashlib.sha256(blob).hexdigest(),
            "n_of": len(raw["entries"]),
            "indices": list(SUBSET_INDICES),
            "built_by": "scripts/noise_v3_harmonicity.py --subsets",
        }
        dst = _subset_path(family)
        dst.write_text(json.dumps(sub))
        print(
            f"{family}: {len(sub['entries'])} of {len(raw['entries'])} -> {dst.relative_to(ROOT)}"
        )


# ---------------------------------------------------------------------------
# the trajectories: the real validation clips
# ---------------------------------------------------------------------------


def _stretch(rps_raw: np.ndarray, n_out: int) -> np.ndarray:
    """``data_processing.streams.stretch_rps_to_frames`` onto ``n_out`` samples:
    endpoint-to-endpoint linear, ``align_corners=False`` — the validation's own
    label alignment, at the audio rate."""
    n_in = int(rps_raw.shape[1])
    pos = np.clip((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0.0, n_in - 1)
    return np.stack([np.interp(pos, np.arange(n_in), row) for row in rps_raw])


def valid_clips() -> tuple[Path, list[dict[str, Any]]]:
    """``(root, clips)``: every validation clip with its labels' speed range and
    whether it is a trajectory (:data:`MIN_CLIP_RPS`)."""
    import soundfile as sf

    from data_processing.streams import resolve_source

    root = Path(resolve_source(VALID))
    meta = {e["id"]: e for e in json.loads((root / "metadata.json").read_text())["valid"]}
    clips = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("sample_")):
        info = sf.info(str(d / "mixture.wav"))
        rps = _stretch(np.load(d / "rps.npy").astype(np.float64), int(info.frames))
        m = meta[d.name]
        clips.append(
            {
                "id": d.name,
                "recording_id": m["recording_id"],
                "source_type": m["source_type"],
                "rig": "michaels" if m["recording_id"].startswith("michaels") else "dregon",
                "start_s": float(m["start_time"]),
                "seconds": round(info.frames / float(info.samplerate), 3),
                "rps_min": [round(float(v), 2) for v in rps.min(axis=1)],
                "rps_max": [round(float(v), 2) for v in rps.max(axis=1)],
                "rps_mean": round(float(rps.mean()), 2),
                "flying": bool(rps.min() >= MIN_CLIP_RPS),
                "loudspeaker": m["source_type"] in LOUDSPEAKER,
            }
        )
    return root, clips


def _clip_io(root: Path, clip_id: str) -> tuple[np.ndarray, np.ndarray]:
    """``(audio (8, T), labels (4, T))`` of one validation clip."""
    import soundfile as sf

    audio, sr = sf.read(str(root / clip_id / "mixture.wav"), dtype="float64", always_2d=True)
    if int(sr) != SR:
        raise ValueError(f"{clip_id}: {sr} Hz, expected {SR}")
    audio = audio.T
    rps = _stretch(np.load(root / clip_id / "rps.npy").astype(np.float64), audio.shape[1])
    return audio, rps


# ---------------------------------------------------------------------------
# the sources
# ---------------------------------------------------------------------------


def source_specs(smoke: bool = False) -> list[dict[str, Any]]:
    """Every rendered source as a plain spec a worker can build."""
    idx = range(len(SUBSET_INDICES))
    specs: list[dict[str, Any]] = []
    for name in ("dregon_cruise_refined", "michael_cruise"):
        specs.append({"family": "legacy_fit", "kind": "LegacyFit", "arg": name})
    for fam, kind, preset in (
        ("legacy_easy", "LegacyBank", "easy"),
        ("legacy_hard", "LegacyBank", "hard"),
    ):
        specs += [{"family": fam, "kind": kind, "arg": preset, "index": i} for i in idx]
    for rig in ("dregon", "michaels"):
        specs.append({"family": "v2_fit", "kind": "V2Fit", "arg": rig})
    for fam, preset in (("v2_easy", "easy"), ("v2_hard", "hard")):
        specs += [{"family": fam, "kind": "V2Bank", "arg": preset, "index": i} for i in idx]
    for rig in ("dregon", "michaels"):
        specs.append({"family": "v3_fit", "kind": "V3Fit", "arg": rig})
    for fam, preset in (("v3_easy", "easy"), ("v3_hard", "hard")):
        specs += [{"family": fam, "kind": "V3Bank", "arg": preset, "index": i} for i in idx]
    for n, s in enumerate(specs):
        s["key"] = f"{s['family']}:{s['arg']}" + (f"[{s['index']}]" if "index" in s else "")
        s["n"] = n
        if "index" in s:
            s["bank_index"] = SUBSET_INDICES[s["index"]]
    if smoke:
        seen: set[str] = set()
        specs = [s for s in specs if not (s["family"] in seen or seen.add(s["family"]))]
    return specs


def _noise_lab() -> Any:
    """``notebooks/noise_lab.py`` with its six bank paths pointed at the subsets.

    The bank dicts are updated IN PLACE: ``V2Bank.BANKS`` / ``V3Bank.BANKS`` hold
    the dict objects themselves. ``V3Bank`` names the round-2 v3 banks by
    default; here it reads the v3r3 subsets, the banks the nv3r3 arms train on.
    """
    if str(ROOT / "notebooks") not in sys.path:
        sys.path.insert(0, str(ROOT / "notebooks"))
    import noise_lab as nl

    rel = {fam: str(_subset_path(fam).relative_to(ROOT)) for fam in BANKS}
    nl.LEGACY_BANKS.update(easy=rel["legacy_easy"], hard=rel["legacy_hard"])
    nl.V2_BANKS.update(easy=rel["v2_easy"], hard=rel["v2_hard"])
    nl.V3_BANKS.update(easy=rel["v3_easy"], hard=rel["v3_hard"])
    return nl


def _build(nl: Any, spec: dict[str, Any]) -> Any:
    kind = spec["kind"]
    if kind == "LegacyFit":
        return nl.LegacyFit(spec["arg"])
    if kind == "LegacyBank":
        return nl.LegacyBank(spec["arg"], spec["index"])
    if kind == "V2Fit":
        return nl.V2Fit(spec["arg"])
    if kind == "V3Fit":
        return nl.V3Fit(spec["arg"], round="r3b")
    if kind == "V2Bank":
        return nl.V2Bank(spec["arg"], spec["index"])
    if kind == "V3Bank":
        return nl.V3Bank(spec["arg"], spec["index"])
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# --run
# ---------------------------------------------------------------------------

_W: dict[str, Any] = {}


def _init(root: str) -> None:
    import torch

    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    _W["root"] = Path(root)
    _W["nl"] = _noise_lab()
    _W["sources"] = {}


def true_carrier(src: Any, labels: np.ndarray) -> np.ndarray:
    """The ``(R, T)`` carrier a render's comb was built on, from the labels it got.

    A v2 / v3 render puts its lines on the track it is handed (its shaft
    wander and line phase noise are ZERO-MEAN processes around it). A legacy
    render does not: ``stochastic_rotor_noise.synthesize`` treats the track as
    the LABEL and rides the comb on ``label + shaft_offset_rps``, a static
    per-rotor error drawn per entry (median 0.6, up to 3.9 rev/s here),
    stopped rotors kept stopped. That offset is the legacy model's label
    error, not its line shape, so the true carrier puts it back, exactly as
    ``synthesize`` does.
    Its OU shaft jitter stays in the render, as the v2 / v3 shaft wander does.
    """
    if getattr(src, "generation", "") != "legacy":
        return labels
    offset = np.asarray(src.params.shaft_offset_rps, dtype=np.float64)
    offset = np.full(labels.shape[0], float(offset)) if offset.ndim == 0 else offset
    shaft = np.maximum(labels + offset[: labels.shape[0], None], 0.0)
    return np.where(labels > 0.0, shaft, 0.0)


def _task(task: tuple[dict[str, Any] | None, int, str]) -> dict[str, Any]:
    from experiments.noise_model import tonality as TN

    spec, clip_n, clip_id = task
    t0 = time.time()
    audio, rps = _clip_io(_W["root"], clip_id)
    carrier = rps
    if spec is None:
        key, entry = "real", clip_id
        x = audio[:N_MICS]
    else:
        key = spec["key"]
        src = _W["sources"].get(key)
        if src is None:
            src = _W["sources"][key] = _build(_W["nl"], spec)
        seed = 1000 * clip_n + int(spec["n"])
        x = np.asarray(src.render(rps, seed=seed, n_mics=N_MICS), dtype=np.float64)
        entry = str(src.entry)
        carrier = true_carrier(src, rps)
    rms = float(np.sqrt(np.mean(np.square(x))))
    t1 = time.time()
    res = TN.demod_prominence(x, carrier, SR)
    row = {
        "key": key,
        "clip": clip_id,
        "entry": entry,
        "prom_db": np.round(res.prom_db, 2).tolist(),
        "floor_db": np.round(res.floor_db, 2).tolist(),
        "n_frames": res.n_frames,
        "finite": bool(np.isfinite(x).all()),
        "rms": rms,
        "render_s": round(t1 - t0, 2),
    }
    if carrier is not rps:
        # the same legacy render read on the LABELS, as a network is trained on it
        row["shaft_offset_rps"] = np.round(np.mean(carrier - rps, axis=1), 3).tolist()
        row["prom_label_db"] = np.round(TN.demod_prominence(x, rps, SR).prom_db, 2).tolist()
    row["metric_s"] = round(time.time() - t1, 2)
    return row


def run(workers: int, smoke: bool) -> None:
    from experiments.noise_model import tonality as TN

    root, clips = valid_clips()
    flying = [c for c in clips if c["flying"]]
    if smoke:
        flying = [c for c in flying if c["rig"] == "michaels"][-1:]
    specs = source_specs(smoke)
    number = {c["id"]: n for n, c in enumerate(clips)}
    tasks = [(None, number[c["id"]], c["id"]) for c in flying]
    tasks += [(s, number[c["id"]], c["id"]) for s in specs for c in flying]
    print(
        f"{len(flying)} flying clips of {len(clips)}; {len(specs)} sources; "
        f"{len(tasks)} tasks on {workers} workers",
        flush=True,
    )
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(str(root),)) as pool:
        for i, row in enumerate(pool.imap_unordered(_task, tasks, chunksize=1)):
            rows.append(row)
            if (i + 1) % 50 == 0 or i + 1 == len(tasks):
                print(f"  {i + 1}/{len(tasks)}  {time.time() - t0:.0f} s", flush=True)
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_key.setdefault(row["key"], {})[row["clip"]] = row
    out = {
        "what": "demodulated prominence (tonality.demod_prominence) per source x clip x rotor x order",
        "built_by": "scripts/noise_v3_harmonicity.py --run" + (" --smoke" if smoke else ""),
        "git_head": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
        "valid": VALID,
        "method": {
            "k_max": TN.DEMOD_K_MAX,
            "lowpass_hz": TN.DEMOD_LOWPASS_HZ,
            "frame_s": TN.DEMOD_FRAME_S,
            "hop_s": TN.DEMOD_HOP_S,
            "ends_frac": list(TN.DEMOD_ENDS_FRAC),
            "n_mics": N_MICS,
            "min_clip_rps": MIN_CLIP_RPS,
            "labels": "rps.npy stretched endpoint-to-endpoint onto the clip (DregonLMFrameDataset)",
            "seed": "1000 * clip number + source number",
        },
        "subset_indices": list(SUBSET_INDICES),
        "clips": clips,
        "sources": [{"key": "real", "family": "real"}] + specs,
        "results": by_key,
        "wall_s": round(time.time() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / ("prominence_smoke.json" if smoke else "prominence.json")
    dst.write_text(json.dumps(out))
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB) in {out['wall_s']} s", flush=True)


# ---------------------------------------------------------------------------
# --plot
# ---------------------------------------------------------------------------


def _stack(
    res: dict[str, Any], keys: list[str], clip_ids: list[str], *, field: str = "prom_db"
) -> np.ndarray:
    """``(n, R, K)`` ``field`` of every (source, clip) present that carries it."""
    rows = [res[k][c] for k in keys for c in clip_ids if c in res.get(k, {})]
    return np.array([r[field] for r in rows if field in r], dtype=np.float64)


def summarise(data: dict[str, Any]) -> dict[str, Any]:
    from experiments.noise_model import tonality as TN

    res = data["results"]
    clips = data["clips"]
    sets = {
        "nosource": [c["id"] for c in clips if c["flying"] and not c["loudspeaker"]],
        "loudspeaker": [c["id"] for c in clips if c["flying"] and c["loudspeaker"]],
    }
    fam_keys: dict[str, list[str]] = {"real": ["real"]}
    for s in data["sources"][1:]:
        fam_keys.setdefault(s["family"], []).append(s["key"])
    out: dict[str, Any] = {
        "what": "demodulated prominence (dB), medians over clips x entries x rotors",
        "source": str(RESULT.relative_to(ROOT)),
        "git_head": data["git_head"],
        "method": data["method"],
        "clip_sets": sets,
        "clips": [c for c in clips if c["flying"]],
        "families": {},
    }
    for fam, keys in fam_keys.items():
        row: dict[str, Any] = {"n_sources": len(keys)}
        for set_name, ids in sets.items():
            p = _stack(res, keys, ids)
            if not p.size:
                continue
            per_k = p.transpose(2, 0, 1).reshape(p.shape[2], -1)  # (K, n * R)
            present = {c for k in keys for c in res.get(k, {}) if c in ids}
            row[set_name] = {
                "n_clips": len(present),
                "n_cells": int(p.shape[0] * p.shape[1]),
                "median_by_k": np.round(np.nanmedian(per_k, axis=1), 2).tolist(),
                "q25_by_k": np.round(np.nanpercentile(per_k, 25, axis=1), 2).tolist(),
                "q75_by_k": np.round(np.nanpercentile(per_k, 75, axis=1), 2).tolist(),
                "groups": TN.order_group_summary(p),
            }
            for rig in ("dregon", "michaels"):
                rid = [c["id"] for c in clips if c["id"] in ids and c["rig"] == rig]
                q = _stack(res, keys, rid)
                if q.size:
                    row[set_name][f"groups_{rig}"] = TN.order_group_summary(q)
            lab = _stack(res, keys, ids, field="prom_label_db")
            if lab.size:
                # a legacy render read on its LABELS (the static shaft offset left in)
                row[set_name]["groups_label"] = TN.order_group_summary(lab)
        out["families"][fam] = row
    return out


def plot(summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = summary["families"]
    real = fams["real"]["nosource"]
    k = np.arange(1, len(real["median_by_k"]) + 1)
    gens = (("legacy", "legacy"), ("v2", "v2"), ("v3", "v3 (r3b fits, v3r3 banks)"))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    for ax, (gen, title) in zip(axes, gens, strict=True):
        ax.fill_between(
            k, real["q25_by_k"], real["q75_by_k"], color="0.8", lw=0, label="real, 25–75 %"
        )
        ax.plot(k, real["median_by_k"], color="0.45", lw=2.0, label="real, median")
        for fam, label, colour in FAMILIES:
            if not fam.startswith(gen + "_") or fam not in fams:
                continue
            row = fams[fam]["nosource"]
            ls = "--" if fam.endswith("_fit") else "-"
            ax.plot(k, row["median_by_k"], ls, color=colour, lw=1.8, label=label)
        for thr in (3.0, 6.0, 10.0):
            ax.axhline(thr, color="0.6", lw=0.6, ls=":")
        ax.set_title(title)
        ax.set_xlabel("order k")
        ax.set_xlim(1, k[-1])
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    axes[0].set_ylabel("prominence, median (dB)")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--subsets", action="store_true", help="cut the 16-entry bank subsets")
    mode.add_argument("--run", action="store_true", help="render + measure (uni-cpu)")
    mode.add_argument("--plot", action="store_true", help="harm_summary.json + figure")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--smoke", action="store_true", help="--run: one source per family, one clip")
    args = ap.parse_args()
    if args.subsets:
        make_subsets()
    elif args.run:
        run(int(args.workers), bool(args.smoke))
    else:
        summary = summarise(json.loads(RESULT.read_text()))
        SUMMARY.write_text(json.dumps(summary, indent=1))
        plot(summary)
        print(f"wrote {SUMMARY.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
