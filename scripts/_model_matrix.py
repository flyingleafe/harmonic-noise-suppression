"""A 24-clip four-model evaluation matrix: real cruise vs three synthetic families.

WHAT THIS ANSWERS. The transfer question is usually asked with one number per
model on one real split, which cannot separate "this model reads rotor speed"
from "this model reads the family it was trained on". So every model here is
scored on FOUR clip families that differ in exactly one thing at a time:

    A  real cruise            3 Michael (FLY124) + 3 DREGON clips of the frozen
                              split, the most cruise-dominated ones available.
    B  matched-RPS synthetic  the SAME six RPS tracks, re-rendered from the
                              corresponding rig's FITTED model. Same labels,
                              same speeds, synthetic audio: the gap A -> B is
                              the audio-realism gap alone.
    C  stream / rig fit       six RPS trajectories from the training stream's
                              OWN generator (`conf/online_mix/rig_easy_5050.yaml`
                              `rps: {kind: full_flight}`), rendered from the two
                              rig fits. New labels, fitted rigs: the gap B -> C
                              is the trajectory-distribution gap.
    D  stream / hard cloud    the SAME six trajectories, each rendered from a
                              random parameter draw of the hard cloud
                              (`data/rig_banks/rig_hard_n2048.json`). The gap
                              C -> D is the rig-family gap, and D is the hard
                              arm's own training distribution.

THE METRIC IS ONE ROTOR ASSIGNMENT PER CLIP, which is what the training loss
(`src/losses/pit.py:94-106`) and the reported metric (`src/metrics/rps.py:65-69`)
both do: the permutation is chosen on the time-averaged cost and then held for
the whole clip. Per-frame Hungarian matching appears NOWHERE in this file — not
in a score, not in a figure. A per-frame match lets a model be scored under one
rotor identity early in a clip and another later, which buys error it did not
earn.

THE FIGURES DRAW RAW OUTPUT. Each model emits four series and four series are
drawn, one colour per model, over the four target tracks. No permutation, ever.
The drawing itself is
`writing/slides/2026-09-15_noise-model-and-fit/prepare_regime.py:draw`, imported
rather than reimplemented, with its module-level `ARMS`/`ASSETS` rebound to this
matrix's four models and output directory.

EVERYTHING EXPENSIVE IS CACHED under `results/model_matrix/`: the six stream
trajectories, the eighteen rendered synthetic clips (as a DREGON-LM-shaped split
so the model sees byte-identical framing to a real clip), the replayed
hard-cloud mixing coordinates, and every model's per-clip prediction. A re-run
re-renders nothing and re-infers nothing whose checkpoint and audio digests
still match; `--refresh` forces the predictions and `--rerender` the audio.

    PYTHONPATH="$PWD/src:$PWD/scripts" python scripts/_model_matrix.py
    PYTHONPATH="$PWD/src:$PWD/scripts" python scripts/_model_matrix.py --n-traj 2
    PYTHONPATH="$PWD/src:$PWD/scripts" python scripts/_model_matrix.py --refresh
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _regime_decomp as rd  # noqa: E402
from valid_regime_eval import VALID, clip_rigs  # noqa: E402

# ─── Output locations (this script owns all three) ────────────────────────────

RESULTS = REPO_ROOT / "results/model_matrix"
FIGURES = REPO_ROOT / "docs/explainers/model-matrix"
RENDER_DIR = RESULTS / "renders"  # a DREGON-LM-shaped split
RENDER_SPLIT = RENDER_DIR / "clips"  # ... whose sample_* dirs live here
TRAJ_DIR = RESULTS / "trajectories"
PRED_DIR = RESULTS / "preds"

# ─── The four models, in the order the table prints them ──────────────────────
#
#: ``(experiment, checkpoint, tag, colour)``. The two HPPNet arms are SALIENCE
#: models: their speeds come from the model's own decoder
#: (`valid_regime_eval.salience_rps_pred`, reached through
#: `_regime_decomp.predict_clip`'s `salience` flag), not from an `rps_pred` head
#: they do not have.
MODELS: tuple[tuple[str, str, str, str], ...] = (
    ("hppnet_l2_r2_s0", "best", "hppnet-real-best", "#111111"),
    ("rig_easy_hppnet_l2_unified", "best_real_overall", "hppnet-easy", "#1f77b4"),
    ("real_r4_scv2_unified", "best_real_overall", "scv2-real-R4", "#2ca02c"),
    ("rig_hard_scv2_unified", "best_real_overall", "scv2-hard", "#d62728"),
)

#: Drawing order, i.e. z-order: the near-black real-data reference goes LAST so
#: it is never hidden under a synthetic arm's line. Scores never depend on this.
DRAW_ORDER = ("hppnet-easy", "scv2-real-R4", "scv2-hard", "hppnet-real-best")

# ─── The clip set ─────────────────────────────────────────────────────────────

#: Clip length of every clip in the matrix, seconds. The frozen real split is
#: 8.0 s per clip (251 frames at 16000/512 Hz), and the synthetic families use
#: the same length so a figure, a frame count and a per-clip mean mean the same
#: thing in all four categories.
DURATION_S = 8.0
SAMPLE_RATE = 16000
N_FFT = 2048
HOP = 512
N_MICS = 8

#: HOW THE CATEGORY-A CLIPS WERE CHOSEN, and it was not by hand. Every clip of
#: the frozen split is labelled by `_regime_decomp.frame_regimes4` and ranked by
#: its CRUISE SHARE; sixteen DREGON clips and seven Michael clips are 100 %
#: cruise, so the share alone does not decide. The stated tie-breaks:
#:
#:   Michael   FLY124 is the split's only Michael recording, so among its seven
#:             100 %-cruise clips take the three with the largest peak-to-peak
#:             excursion of the ROTOR-MEAN speed: a flat clip tests a constant,
#:             a moving one tests tracking.
#:   DREGON    room1 appears under three interferer conditions (nosource,
#:             speech-low, whitenoise-low). Take the most dynamic 100 %-cruise
#:             clip of EACH, so the DREGON half covers all three conditions
#:             instead of three near-copies of one.
#:
#: Filled in by `choose_real_clips`, which recomputes the ranking every run and
#: fails loudly if these indices stop being the winners.
REAL_CHOICE_RULE = {
    "michaels": "3 largest rotor-mean peak-to-peak among the 100%-cruise FLY124 clips",
    "dregon": "most dynamic 100%-cruise clip of each room1 interferer condition",
}

#: The two rig fits every B and C clip is rendered from — the SAME anchors the
#: synthetic training streams use (`conf/online_mix/rig_easy_5050.yaml`'s bank
#: provenance), with the same per-clip dynamics donor.
ANCHORS: dict[str, dict[str, Any]] = {
    "michaels": {
        "fit": "results/S2/cruise_8clip.json",
        "clip": "fly125_cruise_00",
        "dynamics": "conf/online_mix/rig_fitted_5050.yaml:1",
    },
    "dregon": {
        "fit": "results/S2/dregon_room2_cruise_refined.json",
        "clip": "free-flight_nosource_room2_cruise_00",
        "dynamics": "conf/online_mix/rig_fitted_5050.yaml:0",
    },
}

#: The stream whose `rps` block generates the C/D trajectories and whose render
#: settings (8 mics, `line_mode: fm`, mic-gain spread, level draw, n_fft) every
#: synthetic clip is rendered with. Source 0 of its noise list.
STREAM_POLICY = "conf/online_mix/rig_easy_5050.yaml"
STREAM_SOURCE = 0

#: The hard cloud category D draws its rigs from.
HARD_BANK = "data/rig_banks/rig_hard_n2048.json"

#: One seed for the whole matrix; every draw below is a NAMED substream of it,
#: so any single clip is reproducible on its own.
MATRIX_SEED = 20260915
_SUB_FIT = 1  # the per-clip dynamics draw that dresses a rig fit
_SUB_TRAJ = 100  # the stream trajectories
_SUB_RENDER = 200  # the render draw (mic gains, floor GP, phases, level)
_SUB_BANK = 300  # which hard-cloud entries category D uses

#: Above this the projection triggers a cut of the stream trajectories.
BUDGET_MINUTES = 25.0


# ─── Small utilities ──────────────────────────────────────────────────────────


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_digest(payload: Any) -> str:
    """SHA-256 of a JSON payload written canonically."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_jsonable)
    return sha256_bytes(text.encode())


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON-serializable: {type(value)}")


# ─── The donor drawing module ─────────────────────────────────────────────────


def load_prepare_regime(tags_colours: list[tuple[str, str]]) -> Any:
    """Import the deck's `prepare_regime` and rebind it onto THIS matrix.
    The deck's module owns the figure layout, so it is imported and rebound
    rather than reimplemented: `ARMS` becomes this matrix's four models with
    their colours and `ASSETS` becomes this matrix's figure directory. Both are
    module-level constants the drawing code reads, and both are set with
    `setattr` because the module object is untyped.

    The donor module already prints the clip-level convention in its footer, so
    none of its text needs patching here.
    """
    path = REPO_ROOT / "writing/slides/2026-09-15_noise-model-and-fit/prepare_regime.py"
    spec = importlib.util.spec_from_file_location("_mm_prepare_regime", path)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    FIGURES.mkdir(parents=True, exist_ok=True)
    module.ARMS = tuple((tag, tag, colour) for tag, colour in tags_colours)
    module.ASSETS = FIGURES
    return module


# ─── Clips ────────────────────────────────────────────────────────────────────


@dataclass
class Clip:
    """One row of the matrix: what it is, where it came from, and its audio."""

    key: str
    category: str
    rig: str  # michaels | dregon | hard
    group: str  # the rig-split column this clip falls in: michaels | dregon
    label: str  # figure title, left field
    recording: str  # figure title, middle field
    why: str
    provenance: dict[str, Any]
    audio: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    target: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((4, 0)))
    frame: Any = field(repr=False, default=None)
    digest: str = ""


def choose_real_clips(dataset: Any, rows: list[dict], rigs: list[str]) -> list[dict[str, Any]]:
    """The six category-A clips, chosen by :data:`REAL_CHOICE_RULE`, with reasons."""
    ranked: list[dict[str, Any]] = []
    for i in range(len(dataset.samples)):
        target = np.asarray(dataset[i]["rps"].data, dtype=np.float64)
        labels = rd.frame_regimes4(target)
        mean = target.mean(axis=0)
        ranked.append(
            {
                "clip": i,
                "rig": rigs[i],
                "recording": str(rows[i].get("recording_id", "")),
                "sample_id": str(rows[i]["id"]),
                "frames": int(labels.size),
                "cruise_share": float((labels == "cruise").mean()),
                "mean_ptp": float(np.ptp(mean)),
                "mean_speed": float(mean.mean()),
            }
        )

    chosen: list[dict[str, Any]] = []
    michaels = [r for r in ranked if r["rig"] == "michaels"]
    top_share = max(r["cruise_share"] for r in michaels)
    pool = sorted(
        [r for r in michaels if r["cruise_share"] >= top_share - 1e-9],
        key=lambda r: -r["mean_ptp"],
    )
    for rank, row in enumerate(pool[:3], start=1):
        chosen.append(
            dict(
                row,
                why=(
                    f"Michael FLY124, cruise share {row['cruise_share']:.0%} (the split's "
                    f"maximum, shared by {len(pool)} FLY124 clips); rank {rank} of those by "
                    f"rotor-mean peak-to-peak, {row['mean_ptp']:.1f} rev/s about a "
                    f"{row['mean_speed']:.0f} rev/s mean — the most MOVING cruise Michael has"
                ),
            )
        )

    dregon = [r for r in ranked if r["rig"] == "dregon"]
    conditions = sorted({r["recording"] for r in dregon})
    for condition in conditions:
        here = [r for r in dregon if r["recording"] == condition]
        best_share = max(r["cruise_share"] for r in here)
        candidates = sorted(
            [r for r in here if r["cruise_share"] >= best_share - 1e-9],
            key=lambda r: -r["mean_ptp"],
        )
        row = candidates[0]
        chosen.append(
            dict(
                row,
                why=(
                    f"DREGON {condition}, cruise share {row['cruise_share']:.0%}; the most "
                    f"dynamic of that condition's {len(candidates)} top-share clips "
                    f"({row['mean_ptp']:.1f} rev/s peak-to-peak about "
                    f"{row['mean_speed']:.0f} rev/s). One clip per interferer condition so "
                    "the DREGON half is not three copies of one room state"
                ),
            )
        )
    if len(chosen) != 6:
        raise SystemExit(f"category A wants 6 clips, the rules produced {len(chosen)}")
    return chosen


# ─── Rig parameters ───────────────────────────────────────────────────────────


def fitted_rig(rig: str) -> tuple[Any, dict[str, Any]]:
    """The rig fit as the training stream carries it, plus its provenance.

    This is `rig_sampler.entry_params` applied to the ANCHOR ITSELF instead
    of to a neighbourhood draw around it: the fit supplies the rig's identity
    (timbre, floor shape, line widths, per-(mic, rotor) pattern, fitted speed
    law) and the policy's own `ranges:` supply the per-clip dynamics, exactly as
    every bank entry is built. The speed exponents pass through the same bound
    the bank enforces (`clip_exponents`) because a negative floor exponent is
    `0 ** negative` at the exact zero of a full flight's ground phase — infinite
    line power, NaN audio. For DREGON that moves `floor_exp` from -3.71 to 0,
    which is recorded below; at cruise the pool rebases `amp_rps_ref` onto the
    window's own hover, so the floor exponent is close to inert there.
    """
    import _build_rig_bank as bank_builder

    from experiments.stochastic_fit import rig_sampler

    spec = ANCHORS[rig]
    fit_path = REPO_ROOT / spec["fit"]
    anchor = rig_sampler.load_anchor(fit_path, spec["clip"])
    export, clipped = rig_sampler.clip_exponents(anchor)

    policy_text, _, index = str(spec["dynamics"]).rpartition(":")
    ranges = rig_sampler.donor_ranges(REPO_ROOT / policy_text, int(index))
    rates = np.full(
        int(np.atleast_2d(np.asarray(anchor["profile_db"])).shape[0]),
        bank_builder.DEFAULT_RPS_SCALE_MIN * bank_builder.slowest_cruise_rps(),
        dtype=np.float64,
    )
    seed = [MATRIX_SEED, _SUB_FIT, sorted(ANCHORS).index(rig)]
    params = rig_sampler.entry_params(export, ranges, np.random.default_rng(seed), rates=rates)
    provenance = {
        "rig": rig,
        "fit": spec["fit"],
        "fit_sha256": sha256_file(fit_path),
        "anchor_clip": spec["clip"],
        "dynamics_from": spec["dynamics"],
        "dynamics_seed": seed,
        "comb_sized_for_rps": float(rates[0]),
        "exponents_clipped": clipped,
        "amp_rps_exponent": float(params.amp_rps_exponent),
        "amp_rps_exponent_floor": (
            None if params.amp_rps_exponent_floor is None else float(params.amp_rps_exponent_floor)
        ),
        "n_harmonics": int(params.n_harmonics),
        "entry_sha256": None,  # filled below
    }
    import data_processing.stochastic_rotor_noise as srn

    provenance["entry_sha256"] = canonical_digest(srn.params_to_entry(params))
    return params, provenance


def hard_bank_coordinates() -> dict[str, Any]:
    """Every hard-cloud entry's mixing coordinate ``t``, replayed and checked.

    The bank file stores the finished parameter sets and the REALISED
    distribution of ``t``, but not ``t`` per entry — so "which entry, and where
    on the Michael-to-DREGON path it sits" is not readable off the file. It is
    recoverable: `_build_rig_bank.build` gives entry ``i`` its own seed
    substream and the sampler writes ``t`` into `export["_sampler"]`, so
    replaying the accept/reject loop (which depends on the export alone)
    reproduces the exact sequence. The replay is verified against the bank's own
    provenance — the per-anchor entry counts and the recorded mean of ``t`` —
    and cached, because it costs a minute.
    """
    import _build_rig_bank as bank_builder

    from experiments.stochastic_fit import rig_sampler

    bank_path = REPO_ROOT / HARD_BANK
    cache = RESULTS / "hard_bank_coordinates.json"
    bank = json.loads(bank_path.read_text())
    prov = bank["provenance"]
    want = {
        "bank": HARD_BANK,
        "entries_sha256": prov["entries_sha256"],
        "seed": int(prov["seed"]),
        "spread": float(prov["spread"]),
        "max_attempts": int(prov["max_attempts"]),
        "n": int(prov["n"]),
    }
    if cache.is_file():
        cached = json.loads(cache.read_text())
        if {k: cached.get(k) for k in want} == want:
            return cached

    if prov["mode"] != "path":
        raise SystemExit(f"{HARD_BANK}: expected a path-mode cloud, got {prov['mode']!r}")
    anchors = [rig_sampler.load_anchor(REPO_ROOT / a["fit"], a["clip"]) for a in prov["anchors"]]
    levels = (rig_sampler.model_level_db(anchors[0]), rig_sampler.model_level_db(anchors[1]))
    started = time.time()
    coords: list[float] = []
    donors: list[int] = []
    substreams: list[int] = []
    substream = 0
    while len(coords) < want["n"]:
        rng = np.random.default_rng([want["seed"], substream])
        substream += 1
        export = rig_sampler.sample_path(
            anchors[0],
            anchors[1],
            rng,
            spread=want["spread"],
            max_attempts=want["max_attempts"],
        )
        t = float(export["_sampler"]["path"]["t"])
        reference = bank_builder.path_reference(anchors, levels, t)
        if not bank_builder.silent_when_stopped(export):
            continue
        export, clipped = rig_sampler.clip_exponents(export)
        if clipped and not rig_sampler.check_sample(export, reference)["ok"]:
            continue
        coords.append(t)
        donors.append(0 if t < 0.5 else 1)
        substreams.append(substream - 1)

    check = {
        "replayed_mean_t": float(np.mean(coords)),
        "bank_mean_t": float(prov["path_coordinate"]["mean"]),
        "replayed_per_anchor": [donors.count(0), donors.count(1)],
        "bank_per_anchor": [int(a["entries"]) for a in prov["anchors"]],
        "seconds": round(time.time() - started, 1),
    }
    if abs(check["replayed_mean_t"] - check["bank_mean_t"]) > 1e-9 or (
        check["replayed_per_anchor"] != check["bank_per_anchor"]
    ):
        raise SystemExit(
            f"hard-cloud replay does not reproduce the bank's own provenance: {json.dumps(check)}"
        )
    out = dict(
        want,
        bank_sha256=sha256_file(bank_path),
        t=coords,
        donor=donors,
        substream=substreams,
        verified=check,
        anchor_order=[f"{a['fit']}:{a['clip']}" for a in prov["anchors"]],
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    print(
        f"  hard-cloud coordinates: replayed {len(coords)} entries in "
        f"{check['seconds']}s, mean t {check['replayed_mean_t']:.6f} == bank's "
        f"{check['bank_mean_t']:.6f}, per-anchor {check['replayed_per_anchor']} == "
        f"{check['bank_per_anchor']}",
        flush=True,
    )
    return out


# ─── Rendering ────────────────────────────────────────────────────────────────


def stream_source() -> dict[str, Any]:
    """The stochastic source whose `rps` block and render settings we reuse."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / STREAM_POLICY), resolve=True)
    return dict(cfg["sources"]["noise"][STREAM_SOURCE])  # type: ignore[index,call-overload]


def _pinned_pool_class() -> type:
    """`StochasticNoisePool` with the trajectory and the rig both pinned.

    The RENDER PATH IS THE STREAM'S OWN: `render` still rebases `amp_rps_ref`
    onto the window's hover, still scales the linewidths and the shaft jitter
    with the aircraft size, still takes the bank branch that leaves a rig's
    fitted speed law alone, still draws the mic gains and the level, and still
    calls `synthesize` with the policy's `line_mode`/`n_fft`. Only two inputs
    are fixed: WHICH rig (a one-entry bank) and WHICH trajectory.
    """
    import data_processing.stochastic_rotor_noise as srn

    class PinnedPool(srn.StochasticNoisePool):  # type: ignore[misc,valid-type]
        def pin(self, params: Any, rps: np.ndarray, hover: float) -> PinnedPool:
            self._bank = (params,)
            self.preset_bank = "<pinned>"
            self._pinned = (np.asarray(rps, dtype=np.float64), float(hover))
            return self

        def sample_rps(self, rng: Any, duration_s: float) -> np.ndarray:
            rps, hover = self._pinned
            self._hover = hover  # what `render` sizes the comb and widths from
            return rps

    return PinnedPool


def stream_trajectory(index: int, source: dict[str, Any]) -> dict[str, Any]:
    """One RPS trajectory from the stream's own `full_flight` generator, cached.

    A fresh pool per trajectory, so each is an independent flight draw windowed
    once (a shared pool reuses one flight for 32 windows — the training stream's
    economy, not a source of six distinct trajectories).
    """
    import data_processing.stochastic_rotor_noise as srn

    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    cache = TRAJ_DIR / f"traj_{index:02d}.npz"
    seed = [MATRIX_SEED, _SUB_TRAJ, index]
    if cache.is_file():
        blob = np.load(cache)
        if list(blob["seed"]) == seed and float(blob["duration_s"]) == DURATION_S:
            return {
                "index": index,
                "rps": blob["rps"],
                "hover": float(blob["hover"]),
                "seed": seed,
                "path": str(cache.relative_to(REPO_ROOT)),
                "sha256": sha256_file(cache),
            }
    pool = srn.StochasticNoisePool.from_config(
        source, duration_s=DURATION_S, sample_rate=SAMPLE_RATE
    )
    if pool.rps_kind != "full_flight":
        raise SystemExit(f"{STREAM_POLICY} source {STREAM_SOURCE}: rps kind is {pool.rps_kind!r}")
    rps = pool.sample_rps(np.random.default_rng(seed), DURATION_S)
    np.savez(
        cache,
        rps=rps.astype(np.float64),
        hover=np.float64(pool._hover),
        seed=np.asarray(seed),
        duration_s=np.float64(DURATION_S),
    )
    return {
        "index": index,
        "rps": rps,
        "hover": float(pool._hover),
        "seed": seed,
        "path": str(cache.relative_to(REPO_ROOT)),
        "sha256": sha256_file(cache),
    }


def render_synthetic(
    clips: list[dict[str, Any]], source: dict[str, Any], rerender: bool
) -> dict[str, dict[str, Any]]:
    """Render (or reuse) every synthetic clip into a DREGON-LM-shaped split.

    Writing the renders as `sample_*/{mixture.wav,rps.npy}` under a split with a
    `metadata.json` is not bookkeeping for its own sake: it means every clip in
    this matrix — real or synthetic — reaches the models through
    `DregonLMFrameDataset`, so the framing, the mic-0 selection and the
    endpoint-to-endpoint stretch of the label onto the STFT grid are byte-
    identical across categories, and a category difference cannot be an
    input-pipeline difference.
    """
    import soundfile as sf

    import data_processing.stochastic_rotor_noise as srn
    from data_processing.streams import stretch_rps_to_frames

    RENDER_SPLIT.mkdir(parents=True, exist_ok=True)
    manifest_path = RENDER_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    pinned_cls = _pinned_pool_class()
    source_digest = canonical_digest(source)

    out: dict[str, dict[str, Any]] = {}
    metadata: list[dict[str, Any]] = []
    for position, spec in enumerate(clips):
        sample_id = f"sample_{position:04d}"
        directory = RENDER_SPLIT / sample_id
        rps_audio = np.asarray(spec["rps_audio"], dtype=np.float64)
        render_seed = [MATRIX_SEED, _SUB_RENDER, position]
        inputs = {
            "key": spec["key"],
            "duration_s": DURATION_S,
            "sample_rate": SAMPLE_RATE,
            "n_mics": N_MICS,
            "render_seed": render_seed,
            "stream_source": source_digest,
            "params": srn.params_to_entry(spec["params"]),
            "rps_sha256": sha256_bytes(rps_audio.tobytes()),
            "hover": spec["hover"],
        }
        digest = canonical_digest(inputs)
        cached = manifest.get(sample_id)
        wav, npy = directory / "mixture.wav", directory / "rps.npy"
        if (
            not rerender
            and cached
            and cached.get("inputs_digest") == digest
            and wav.is_file()
            and npy.is_file()
        ):
            record = dict(cached)
            record["rendered"] = False
        else:
            directory.mkdir(parents=True, exist_ok=True)
            pool = pinned_cls.from_config(
                source, duration_s=DURATION_S, sample_rate=SAMPLE_RATE
            ).pin(spec["params"], rps_audio, spec["hover"])
            started = time.time()
            audio, rps_out, params, _diag = pool.render(
                np.random.default_rng(render_seed), DURATION_S
            )
            seconds = time.time() - started
            if not np.isfinite(audio).all():
                raise SystemExit(f"{spec['key']}: render produced non-finite audio")
            n_frames = audio.shape[-1] // HOP + 1
            frames = stretch_rps_to_frames(rps_audio.astype(np.float32), n_frames)
            # 32-bit float wav: the rig model's floor sits far below its comb
            # peaks and 16-bit PCM would quantize the quiet end of the very
            # spectrum a model has to read.
            sf.write(wav, audio.T.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
            np.save(npy, frames.astype(np.float32))
            record = {
                "key": spec["key"],
                "inputs_digest": digest,
                "render_seed": render_seed,
                "seconds": round(seconds, 2),
                "peak": float(np.abs(audio).max()),
                "rms_mic0": float(np.sqrt(np.mean(audio[0] ** 2))),
                "n_frames": int(n_frames),
                "hover_rev_s": float(spec["hover"]),
                "wav_sha256": sha256_file(wav),
                "rendered": True,
            }
        record["sample_id"] = sample_id
        manifest[sample_id] = {k: v for k, v in record.items() if k != "rendered"}
        out[spec["key"]] = record
        metadata.append(
            {
                "id": sample_id,
                "recording_id": spec["recording"],
                "matrix_key": spec["key"],
                "n_channels": N_MICS,
                "duration": DURATION_S,
                "is_real_recording": False,
            }
        )
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    (RENDER_DIR / "metadata.json").write_text(json.dumps({RENDER_SPLIT.name: metadata}, indent=1))
    return out


# ─── Building the 24 clips ────────────────────────────────────────────────────


def build_clips(n_traj: int, rerender: bool) -> tuple[list[Clip], dict[str, Any]]:
    """Every clip of the matrix, in category order, with its audio loaded."""
    import data_processing.stochastic_rotor_noise as srn
    from data_processing.frame_datasets import DregonLMFrameDataset
    from data_processing.streams import ensure_local, stretch_rps_to_frames

    real = DregonLMFrameDataset(
        data_dir=VALID, n_fft=N_FFT, hop_length=HOP, sample_rate=SAMPLE_RATE, channel=0
    )
    rows = json.loads(
        (Path(ensure_local(VALID.removeprefix("dload:"))) / "metadata.json").read_text()
    )
    if isinstance(rows, dict):
        rows = next(iter(rows.values()))
    chosen = choose_real_clips(real, rows, clip_rigs())

    clips: list[Clip] = []
    for row in chosen:
        i = int(row["clip"])
        frame = real[i]
        audio = np.asarray(frame["mixture"].data, dtype=np.float64)
        target = np.asarray(frame["rps"].data, dtype=np.float64)
        clips.append(
            Clip(
                key=f"A_real_{row['rig']}_clip{i:02d}",
                category="A",
                rig=row["rig"],
                group=row["rig"],
                label=f"A real cruise · {row['rig']}",
                recording=f"{row['recording']} {row['sample_id']}",
                why=str(row["why"]),
                provenance={
                    "source": "frozen real split",
                    "split": VALID,
                    "split_version": rd.split_version(),
                    "clip_index": i,
                    "sample_id": row["sample_id"],
                    "recording_id": row["recording"],
                    "cruise_share": row["cruise_share"],
                    "mean_rotor_ptp_rev_s": row["mean_ptp"],
                    "channel": 0,
                },
                audio=audio,
                target=target,
                frame=frame,
                digest=sha256_bytes(audio.astype(np.float32).tobytes()),
            )
        )

    fits = {rig: fitted_rig(rig) for rig in ("michaels", "dregon")}
    hard = hard_bank_coordinates()
    bank_entries = srn.load_preset_bank(REPO_ROOT / HARD_BANK)

    # Category B: the six real RPS tracks, re-rendered from the matching fit.
    synthetic: list[dict[str, Any]] = []
    for clip in clips:
        rig = clip.rig
        params, fit_prov = fits[rig]
        target = clip.target
        n_samples = int(round(DURATION_S * SAMPLE_RATE))
        # Up onto the audio grid with the same endpoint-to-endpoint stretch the
        # dataset uses to come back down, so the label the model is scored
        # against is the real clip's own track to the last frame.
        rps_audio = stretch_rps_to_frames(target.astype(np.float32), n_samples).astype(np.float64)
        synthetic.append(
            {
                "key": clip.key.replace("A_real_", "B_matched_"),
                "category": "B",
                "rig": rig,
                "group": rig,
                "label": f"B matched-RPS · {rig} fit",
                "recording": f"{rig} fit on {clip.provenance['recording_id']} "
                f"{clip.provenance['sample_id']} RPS",
                "params": params,
                "rps_audio": rps_audio,
                "hover": max(float(np.percentile(target, 90.0)), 1.0),
                "target_frames": target,
                "why": (
                    f"the {rig} rig's FITTED model driven by real clip "
                    f"{clip.provenance['clip_index']}'s own RPS track: same labels as the A "
                    "clip above it, synthetic audio, so the A-to-B gap is audio realism "
                    "alone"
                ),
                "provenance": {
                    "source": "fitted rig render on a real RPS track",
                    "rps_from": {
                        "split": VALID,
                        "clip_index": clip.provenance["clip_index"],
                        "sample_id": clip.provenance["sample_id"],
                        "recording_id": clip.provenance["recording_id"],
                    },
                    "rig_fit": fit_prov,
                },
            }
        )

    # Categories C and D: the stream's own trajectories, twice over.
    source = stream_source()
    fit_for_traj = ["michaels", "michaels", "michaels", "dregon", "dregon", "dregon"]
    bank_rng = np.random.default_rng([MATRIX_SEED, _SUB_BANK, 0])
    bank_choice = [int(i) for i in bank_rng.choice(len(bank_entries), size=6, replace=False)]
    for k in range(n_traj):
        traj = stream_trajectory(k, source)
        rig = fit_for_traj[k]
        params, fit_prov = fits[rig]
        traj_prov = {
            "source": "the training stream's own full_flight generator",
            "policy": f"{STREAM_POLICY}:sources.noise[{STREAM_SOURCE}].rps",
            "kind": "full_flight",
            "generator": "data_processing.stochastic_rotor_noise."
            "StochasticNoisePool.sample_rps -> rps_synthesis.generate_full_flight",
            "trajectory_index": k,
            "trajectory_seed": traj["seed"],
            "trajectory_cache": traj["path"],
            "trajectory_sha256": traj["sha256"],
            "hover_rev_s": traj["hover"],
            "speed_range_rev_s": [float(traj["rps"].min()), float(traj["rps"].max())],
        }
        synthetic.append(
            {
                "key": f"C_stream_{rig}fit_t{k}",
                "category": "C",
                "rig": rig,
                "group": rig,
                "label": f"C stream traj · {rig} fit",
                "recording": f"trajectory {k} · {rig} fitted rig",
                "params": params,
                "rps_audio": traj["rps"],
                "hover": traj["hover"],
                "target_frames": None,
                "why": (
                    f"stream trajectory {k} (seed {traj['seed']}) rendered from the {rig} "
                    "FITTED rig: new labels from the training generator, a rig the arms were "
                    "fitted against, so the B-to-C gap is the trajectory distribution"
                ),
                "provenance": {**traj_prov, "rig_fit": fit_prov},
            }
        )
        entry_index = bank_choice[k]
        entry = bank_entries[entry_index]
        synthetic.append(
            {
                "key": f"D_stream_hard_t{k}",
                "category": "D",
                "rig": "hard",
                "group": rig,
                "label": "D stream traj · hard cloud",
                "recording": f"trajectory {k} · hard entry {entry_index} (t={hard['t'][entry_index]:.3f})",
                "params": entry,
                "rps_audio": traj["rps"],
                "hover": traj["hover"],
                "target_frames": None,
                "why": (
                    f"stream trajectory {k} again, rendered from hard-cloud entry "
                    f"{entry_index} at mixing coordinate t={hard['t'][entry_index]:.3f} "
                    f"(nearer the {['michaels', 'dregon'][hard['donor'][entry_index]]} anchor "
                    "for its dynamics). The C twin above holds the trajectory fixed, so the "
                    "C-to-D gap is the rig family — and D is the hard arm's own training "
                    "distribution"
                ),
                "provenance": {
                    **traj_prov,
                    "rig_source": {
                        "bank": HARD_BANK,
                        "bank_sha256": hard["bank_sha256"],
                        "entries_sha256": hard["entries_sha256"],
                        "entry_index": entry_index,
                        "mixing_coordinate_t": hard["t"][entry_index],
                        "path_anchor_order": hard["anchor_order"],
                        "dynamics_donor_anchor": hard["anchor_order"][hard["donor"][entry_index]],
                        "builder_substream": hard["substream"][entry_index],
                        "choice_seed": [MATRIX_SEED, _SUB_BANK, 0],
                        "choice_rule": "6 distinct entries, uniform without replacement",
                        "amp_rps_exponent": float(entry.amp_rps_exponent),
                        "amp_rps_exponent_floor": (
                            None
                            if entry.amp_rps_exponent_floor is None
                            else float(entry.amp_rps_exponent_floor)
                        ),
                    },
                },
            }
        )

    rendered = render_synthetic(synthetic, source, rerender)

    synth_ds = DregonLMFrameDataset(
        data_dir=RENDER_SPLIT, n_fft=N_FFT, hop_length=HOP, sample_rate=SAMPLE_RATE, channel=0
    )
    by_sample = {p.name: i for i, p in enumerate(synth_ds.samples)}
    for spec in synthetic:
        record = rendered[spec["key"]]
        frame = synth_ds[by_sample[record["sample_id"]]]
        audio = np.asarray(frame["mixture"].data, dtype=np.float64)
        target = np.asarray(frame["rps"].data, dtype=np.float64)
        clips.append(
            Clip(
                key=spec["key"],
                category=spec["category"],
                rig=spec["rig"],
                group=spec["group"],
                label=spec["label"],
                recording=spec["recording"],
                why=spec["why"],
                provenance={
                    **spec["provenance"],
                    "render": {
                        "sample": str((RENDER_SPLIT / record["sample_id"]).relative_to(REPO_ROOT)),
                        "render_seed": record["render_seed"],
                        "wav_sha256": record["wav_sha256"],
                        "peak": record["peak"],
                        "rms_mic0": record["rms_mic0"],
                        "n_mics": N_MICS,
                        "sample_rate": SAMPLE_RATE,
                        "n_fft": N_FFT,
                        "hop_length": HOP,
                        "duration_s": DURATION_S,
                        "level_mode": "window",
                        "normalize_rms": 0.1,
                        "speech_or_interferer": "none: rotor noise only",
                        "render_path": "StochasticNoisePool.render (bank branch) -> "
                        "stochastic_rotor_noise.synthesize",
                        "stream_source": f"{STREAM_POLICY}:sources.noise[{STREAM_SOURCE}]",
                    },
                },
                audio=audio,
                target=target,
                frame=frame,
                digest=sha256_bytes(audio.astype(np.float32).tobytes()),
            )
        )

    order = {"A": 0, "B": 1, "C": 2, "D": 3}
    clips.sort(key=lambda c: (order[c.category], c.key))
    shared = {
        "fits": {rig: prov for rig, (_p, prov) in fits.items()},
        "hard_bank": {
            k: hard[k] for k in ("bank", "bank_sha256", "entries_sha256", "seed", "n", "verified")
        },
        "stream_source": {
            "policy": STREAM_POLICY,
            "source_index": STREAM_SOURCE,
            "sha256": sha256_file(REPO_ROOT / STREAM_POLICY),
            "digest": canonical_digest(source),
            "rps": source.get("rps"),
        },
        "renders": {k: {kk: vv for kk, vv in v.items()} for k, v in rendered.items()},
    }
    return clips, shared


# ─── Scoring ──────────────────────────────────────────────────────────────────


def pit_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """The reported metric: ONE rotor assignment for the whole clip."""
    import torch

    from metrics.rps import batched_pit_mae

    return float(
        batched_pit_mae(
            torch.as_tensor(pred[None], dtype=torch.float32),
            torch.as_tensor(target[None], dtype=torch.float32),
        )[0]
    )


def predict(
    tag: str, model: Any, salience: bool, clip: Clip, ckpt_digest: str, refresh: bool
) -> tuple[np.ndarray, float, bool]:
    """``(pred, seconds, from_cache)`` — cached on the checkpoint and audio digests."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    cache = PRED_DIR / f"{tag}__{clip.key}.npz"
    if cache.is_file() and not refresh:
        blob = np.load(cache, allow_pickle=False)
        if str(blob["ckpt"]) == ckpt_digest and str(blob["clip"]) == clip.digest:
            return blob["pred"], 0.0, True
    started = time.time()
    pred = rd.predict_clip(model, clip.frame, salience)
    seconds = time.time() - started
    np.savez(cache, pred=pred, ckpt=np.array(ckpt_digest), clip=np.array(clip.digest))
    return pred, seconds, False


# ─── The table ────────────────────────────────────────────────────────────────

CATEGORY_TITLE = {
    "A": "A real cruise",
    "B": "B matched-RPS",
    "C": "C stream/rig-fit",
    "D": "D stream/hard",
}


def _agg(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "median": None, "n": 0}
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "n": len(values),
    }


def aggregate(clips: list[Clip], scores: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Per model, per category: the aggregate and the michaels/dregon split."""
    out: dict[str, Any] = {}
    for _exp, _ckpt, tag, _colour in MODELS:
        per_category: dict[str, Any] = {}
        for category in ("A", "B", "C", "D"):
            here = [c for c in clips if c.category == category]
            cell: dict[str, Any] = {"all": _agg([scores[tag][c.key] for c in here])}
            for group in ("michaels", "dregon"):
                members = [c for c in here if c.group == group]
                if members:
                    cell[group] = _agg([scores[tag][c.key] for c in members])
            cell["per_clip"] = {c.key: scores[tag][c.key] for c in here}
            cell["worst_clip"] = max(here, key=lambda c: scores[tag][c.key]).key if here else None
            per_category[category] = cell
        out[tag] = per_category
    return out


def table(clips: list[Clip], scores: dict[str, dict[str, float]], stat: str) -> str:
    """The deliverable table: models down, categories across, rig split inside."""
    groups = ("michaels", "dregon", "all")
    header1 = f"{'':<20}" + "".join(f"  {CATEGORY_TITLE[c]:^22}" for c in ("A", "B", "C", "D"))
    header2 = f"{'model':<20}" + "".join(
        "  " + "".join(f"{g[:4]:>7}" for g in groups) + " " for _ in range(4)
    )
    lines = [header1, header2, "-" * len(header2)]
    agg = aggregate(clips, scores)
    for _exp, _ckpt, tag, _colour in MODELS:
        row = f"{tag:<20}"
        for category in ("A", "B", "C", "D"):
            cell = agg[tag][category]
            row += "  "
            for group in groups:
                value = cell.get(group, {}).get(stat)
                row += f"{value:>7.2f}" if value is not None else f"{'--':>7}"
            row += " "
        lines.append(row)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-traj", type=int, default=6, help="stream trajectories for C and D")
    ap.add_argument("--refresh", action="store_true", help="re-run inference, ignoring the cache")
    ap.add_argument("--rerender", action="store_true", help="re-render the synthetic audio")
    ap.add_argument("--no-figures", action="store_true", help="scores and table only")
    args = ap.parse_args()

    import torch

    import zoo

    torch.set_num_threads(max(1, torch.get_num_threads()))
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    wall = time.time()

    print("loading the four checkpoints", flush=True)
    models: dict[str, tuple[Any, bool]] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    for experiment, ckpt, tag, _colour in MODELS:
        model = zoo.load(experiment, ckpt=ckpt, device="cpu")
        salience = bool(getattr(getattr(model, "model", None), "outputs_salience", False))
        models[tag] = (model, salience)
        uri = rd.checkpoint_uri(experiment, ckpt)
        checkpoints[tag] = {
            "experiment": experiment,
            "ckpt": ckpt,
            "uri": uri,
            "digest": rd.checkpoint_digest(uri),
            "salience_decoder": salience,
            "recorded_training_score": rd.recorded_training_score(experiment),
        }
        print(f"  {tag:<20} {experiment:<28} {ckpt:<18} salience={salience}", flush=True)

    # ─ the projection, measured before anything is rendered, so a cut to the
    # stream trajectories costs nothing that has already been paid for.
    probe_clips, _ = build_clips(0, args.rerender)
    probe = probe_clips[0]
    print(f"\ntiming the grid on {probe.key}", flush=True)
    probe_seconds: dict[str, float] = {}
    probe_preds: dict[str, np.ndarray] = {}
    for _exp, _ckpt, tag, _colour in MODELS:
        model, salience = models[tag]
        pred, seconds, cached = predict(
            tag, model, salience, probe, checkpoints[tag]["digest"], args.refresh
        )
        if cached:  # a cache hit measures nothing; time it for real
            started = time.time()
            pred = rd.predict_clip(model, probe.frame, salience)
            seconds = time.time() - started
        probe_seconds[tag] = seconds
        probe_preds[tag] = pred
        print(f"  {tag:<20} {seconds:6.2f} s", flush=True)
    first_tag = MODELS[0][2]
    per_clip = sum(probe_seconds.values())
    n_traj = max(0, int(args.n_traj))
    projection = (6 + 6 + 2 * n_traj) * per_clip / 60.0
    print(
        f"\nPROJECTION  first model ({first_tag}) {probe_seconds[first_tag]:.1f} s on the first "
        f"clip; all four models {per_clip:.1f} s per clip\n"
        f"            {6 + 6 + 2 * n_traj} clips x 4 models = {projection:.1f} min of inference "
        f"(+ ~{2 * n_traj + 6} renders at ~1.3 s and 24 figures at ~1.5 s)",
        flush=True,
    )
    if projection > BUDGET_MINUTES and n_traj > 0:
        allowed = max(1, int((BUDGET_MINUTES * 60.0 / per_clip - 12) // 2))
        print(
            f"            OVER the {BUDGET_MINUTES:.0f} min budget: CUTTING the stream "
            f"trajectories from {n_traj} to {allowed} (categories C and D lose "
            f"{2 * (n_traj - allowed)} clips; A and B are untouched)",
            flush=True,
        )
        n_traj = allowed
    else:
        print(f"            within the {BUDGET_MINUTES:.0f} min budget, nothing cut", flush=True)

    print("\nbuilding the clip set", flush=True)
    clips, shared = build_clips(n_traj, args.rerender)
    for clip in clips:
        print(f"  {clip.category} {clip.key:<34} {clip.recording}", flush=True)

    print("\nscoring", flush=True)
    scores: dict[str, dict[str, float]] = {tag: {} for _e, _c, tag, _col in MODELS}
    preds: dict[str, dict[str, np.ndarray]] = {tag: {} for _e, _c, tag, _col in MODELS}
    widths: dict[str, int] = {}
    hits = 0
    for clip in clips:
        width = clip.target.shape[1]
        for _exp, _ckpt, tag, _colour in MODELS:
            model, salience = models[tag]
            if clip.key == probe.key and not args.refresh:
                pred, cached = probe_preds[tag], True
            else:
                pred, _seconds, cached = predict(
                    tag, model, salience, clip, checkpoints[tag]["digest"], args.refresh
                )
            hits += int(cached)
            preds[tag][clip.key] = pred
            width = min(width, pred.shape[1])
        widths[clip.key] = width
        for _exp, _ckpt, tag, _colour in MODELS:
            pred = preds[tag][clip.key][:, :width]
            preds[tag][clip.key] = pred
            scores[tag][clip.key] = pit_mae(pred, clip.target[:, :width])
        best = min(MODELS, key=lambda m: scores[m[2]][clip.key])[2]
        print(
            f"  {clip.key:<34} "
            + "  ".join(f"{tag} {scores[tag][clip.key]:6.2f}" for _e, _c, tag, _col in MODELS)
            + f"   best: {best}",
            flush=True,
        )
    print(f"  ({hits} of {len(clips) * len(MODELS)} predictions came from the cache)", flush=True)

    # ─ figures
    notes: list[dict[str, Any]] = []
    if not args.no_figures:
        print("\ndrawing", flush=True)
        colours = {tag: colour for _e, _c, tag, colour in MODELS}
        pr = load_prepare_regime([(tag, colours[tag]) for tag in DRAW_ORDER])
        for clip in clips:
            width = widths[clip.key]
            temp, note = pr.draw(
                clip.key,
                clip.recording,
                clip.label,
                clip.audio,
                clip.target[:, :width],
                {tag: preds[tag][clip.key] for tag in DRAW_ORDER},
                {tag: scores[tag][clip.key] for tag in DRAW_ORDER},
                {},
                clip.why,
            )
            final = FIGURES / f"{clip.key}.png"
            Path(temp).replace(final)
            notes.append(
                {
                    "figure": final.name,
                    "category": clip.category,
                    "key": clip.key,
                    "rig": clip.rig,
                    "rig_split_group": clip.group,
                    "why": clip.why,
                    "pit_mae": {tag: scores[tag][clip.key] for _e, _c, tag, _col in MODELS},
                    "frames": int(width),
                    "duration_s": float(clip.audio.size / SAMPLE_RATE),
                    "provenance": clip.provenance,
                    "draw_note": note.replace(str(temp.name), final.name),
                }
            )
            print(
                f"  {final.relative_to(REPO_ROOT)}  {final.stat().st_size / 1e3:.0f} kB", flush=True
            )
        (FIGURES / "index.json").write_text(
            json.dumps(
                {
                    "generated_by": "scripts/_model_matrix.py",
                    "git_head": rd.git_head(),
                    "metric": "PIT MAE, one rotor assignment per clip "
                    "(metrics.rps.batched_pit_mae); mic 0",
                    "drawing": "writing/slides/2026-09-15_noise-model-and-fit/"
                    "prepare_regime.py:draw, model tracks RAW (no permutation)",
                    "models": {
                        tag: {"colour": colour, **checkpoints[tag]}
                        for _e, _c, tag, colour in MODELS
                    },
                    "draw_order": list(DRAW_ORDER),
                    "figures": notes,
                },
                indent=1,
                default=_jsonable,
            )
        )

    # ─ the deliverable
    print("\n" + "=" * 118)
    print(
        "PIT MAE (rev/s), ONE rotor assignment per clip, mic 0 — MEAN over the clips of each cell"
    )
    print("=" * 118)
    mean_table = table(clips, scores, "mean")
    print(mean_table)
    print()
    print("same cells, MEDIAN over clips")
    median_table = table(clips, scores, "median")
    print(median_table)
    print(
        "\nmich/dreg = the rig the clip belongs to: the recording for A, the rig fit that "
        "rendered it for B and C,\nand for D the trajectory half whose C twin used that fit "
        "(a D clip's own rig is a hard-cloud draw)."
    )

    agg = aggregate(clips, scores)
    loud: list[str] = []
    for _e, _c, tag, _col in MODELS:
        for category in ("A", "B", "C", "D"):
            cell = agg[tag][category]["all"]
            if cell["n"] and cell["max"] >= 3.0 * max(cell["mean"], 1e-9):
                loud.append(
                    f"  {tag} on {CATEGORY_TITLE[category]}: worst clip "
                    f"{agg[tag][category]['worst_clip']} at {cell['max']:.2f} against a cell "
                    f"mean of {cell['mean']:.2f} — the mean hides it"
                )
            if cell["n"] and cell["mean"] >= 20.0:
                loud.append(
                    f"  {tag} on {CATEGORY_TITLE[category]}: cell mean {cell['mean']:.2f} rev/s "
                    "— this is a FAILURE, not a score"
                )
    if loud:
        print("\nLOUD WARNINGS")
        print("\n".join(loud))

    payload = {
        "generated_by": "scripts/_model_matrix.py",
        "git_head": rd.git_head(),
        "metric": {
            "name": "PIT MAE (rev/s)",
            "assignment": "ONE rotor permutation per clip, chosen on the time-averaged "
            "absolute error",
            "implementation": "metrics.rps.batched_pit_mae",
            "matches": ["src/losses/pit.py:94-106", "src/metrics/rps.py:65-69"],
            "per_frame_matching_used": False,
            "channels": "mic 0 only",
        },
        "clip_set": {
            "n_clips": len(clips),
            "duration_s": DURATION_S,
            "sample_rate": SAMPLE_RATE,
            "n_fft": N_FFT,
            "hop_length": HOP,
            "n_mics_rendered": N_MICS,
            "stream_trajectories": n_traj,
            "real_choice_rule": REAL_CHOICE_RULE,
            "split": VALID,
            "split_version": rd.split_version(),
        },
        "seeds": {
            "matrix_seed": MATRIX_SEED,
            "substreams": {
                "rig_fit_dynamics": _SUB_FIT,
                "stream_trajectory": _SUB_TRAJ,
                "render": _SUB_RENDER,
                "hard_bank_choice": _SUB_BANK,
            },
        },
        "models": {tag: checkpoints[tag] for _e, _c, tag, _col in MODELS},
        "provenance": shared,
        "clips": [
            {
                "key": c.key,
                "category": c.category,
                "rig": c.rig,
                "rig_split_group": c.group,
                "recording": c.recording,
                "why": c.why,
                "frames_scored": widths[c.key],
                "audio_sha256": c.digest,
                "pit_mae": {tag: scores[tag][c.key] for _e, _c2, tag, _col in MODELS},
                "provenance": c.provenance,
            }
            for c in clips
        ],
        "aggregates": agg,
        "tables": {"mean": mean_table, "median": median_table},
        "warnings": loud,
        "wall_seconds": round(time.time() - wall, 1),
    }
    (RESULTS / "matrix.json").write_text(json.dumps(payload, indent=1, default=_jsonable))
    print(
        f"\nwrote {(RESULTS / 'matrix.json').relative_to(REPO_ROOT)} in {time.time() - wall:.0f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
