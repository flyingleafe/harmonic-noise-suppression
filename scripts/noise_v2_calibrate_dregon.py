#!/usr/bin/env python
"""R5 DREGON calibration: the ONE comb offset that crosses the trackers' ON gate.

The R5 probe (`results/noise_v2/rounds/round5/tracker_probe/findings.md`) found
that what the two RPS trackers do with a DREGON-like render is a BINARY verdict
bought with one scalar: the carrier gain `S8(1) - med_{alpha != 1} S8(alpha)`
read on HPPNet's OWN front end (`CQT2010v2`, 48 bins/octave, hop 512). Every
render above roughly **5.6 dB** of it is tracked; the one below (R3's
`flight_floor_lowk` fit to the real clip, +1.63 dB) is called SILENCE and
scores 69.6 rev/s. The threshold saturates, so it is not a quantity a Whittle
likelihood can trade against continuously: an objective term would fight the
data for a step function.

This script therefore does NOT fit. It takes the R5 `flight_profile` fit as it
stands, renders it on the three frozen cruise score windows through the widen
runner's own route (`noise_v2_widen_dregon.mutate` + `render.render_noise`, seed
2001, 8 microphones), and BISECTS one scalar dB offset applied to `profile_db`
of every rotor and every order — the semantics of `comb_gain_db`, which the
model folds into the profile and the renderer therefore reads already applied —
to the SMALLEST offset whose carrier gain clears the threshold on EVERY window.
The pin is then PRICED: the same rig is scored by the Whittle objective on the
THREE SCORE WINDOWS' own cells — the windows the pin is bisected on — and the
difference in nats per cell is what the pin costs in likelihood. Nothing here
is a fit and nothing here is free: one scalar, one threshold, one price.

The price is NOT taken on the fit's own five 8 s segments. That batch's
forward model is 255 frames of a 4-rotor, 88-order comb on an 8192-point work
grid and needs over 20 GB of intermediates; it belongs on a 64 GB node and the
command that runs it there is `FIT_POOL_COST_COMMAND` (`--pool fit
--allow-fit-pool`). On the score windows the objective is evaluated in frame
chunks — the Whittle risk is a weighted SUM over frames, so chunking is exact
— and the whole step stays under a gigabyte.

    # bisect the offset, price it, write the pinned rig as a fit JSON
    python scripts/noise_v2_calibrate_dregon.py pin \
        --fit results/noise_v2/rounds/round5/fits/dregon_room2_floor__flight_profile.json
    # score HPPNet + SCv2 on the three windows, as-is, pinned and pinned +3 dB
    python scripts/noise_v2_calibrate_dregon.py score
    # the figures
    python scripts/noise_v2_calibrate_dregon.py figures

`pin` is CPU-cheap (a render plus a CQT is ~4 s per window and offset);
`score` runs the two trackers and is the slow step.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.noise_model import render as RD  # noqa: E402

SCHEMA = "noise-v2-dregon-calibration/1"
OUT_DEFAULT = Path("results/noise_v2/rounds/round5/calibration")
FIT_DEFAULT = Path("results/noise_v2/rounds/round5/fits/dregon_room2_floor__flight_profile.json")
#: The pinned rig, written as a fit JSON so the round scorer and the widen
#: runner can render it through their own unchanged `--fit` paths.
PIN_STEM = "dregon_room2_floor__flight_profile_pin"

#: The ON side of the boundary the probe measured: `v2_real` +1.63 dB is OFF
#: (69.6 rev/s), `v2_legacy_free` +5.57, `legacy` +5.69 and `v2_legacy_lowk`
#: +5.98 are ON (1.75 / 1.96 / 5.56 rev/s). 5.6 dB is the lowest ON arm,
#: rounded to the reported precision.
S8_THRESHOLD = 5.6
#: The search: an offset can only RAISE the comb, 24 dB is well past the
#: level-matched shift the R2 registration study measured (+21.8 dB), and
#: 0.25 dB is a tenth of the spread between the ON arms.
OFFSET_LO, OFFSET_HI, OFFSET_STEP = 0.0, 24.0, 0.25
#: A coarse sweep evaluated alongside the bisection, so the monotonicity the
#: bisection assumes is EVIDENCE in the record rather than an assumption.
AUDIT_STEP = 3.0
#: Orders the prominence ladder is reported at (the probe's `c_k`, the read
#: bin over the inter-tooth floor).
LADDER_ORDERS = tuple(range(1, 9))
#: The support set both halves of the R5 pool live in: the five 4 s SCORED
#: windows and the five disjoint 8 s FLOOR segments the fit itself ran on.
SUPPORT_SET = "dregon-floor"
#: Frame stride of the priced batch, the fit's own
#: (`diagnostics.batch.frame_stride`); this is the fallback when a payload
#: does not carry one. At hop 512 and NFFT 2048 a stride of 4 selects
#: DISJOINT windows.
SCORE_FRAME_STRIDE = 4
#: Frames per forward pass when the objective is evaluated. The flight
#: forward model holds `(R, n_c, K, n_fft_work)` intermediates — 4 rotors, a
#: 32-order harmonic chunk and an 8192-point work grid, i.e. ~8 MB per frame
#: per temporary — so the fit's own 255-frame pool needs over 20 GB and was
#: OOM-killed three times on this laptop. The Whittle risk is a WEIGHTED SUM
#: over frames (`revised_phase.composite_risk`), so evaluating it in frame
#: chunks and adding the chunks is EXACT, not an approximation.
WHITTLE_FRAME_CHUNK = 8
#: `noise_v2_fit.py flight --max-frames`, the value every DREGON flight fit
#: of this campaign ran with; the `--pool fit` batch is checked against the
#: frame count the payload recorded.
FIT_MAX_FRAMES = 256
#: The price on the FIT'S OWN five 8 s segments is a CLUSTER job, never a
#: laptop one: the pool is three times the score windows' frames and the
#: laptop rule for this step is that the fit pool's forward model is never
#: built here (three OOM kills bought that rule). This is the command that
#: computes it, verbatim; `--allow-fit-pool` is the switch that lets the
#: script build that pool at all.
FIT_POOL_COST_COMMAND = (
    "omnirun --daemon localhost:18787 submit --backend uni-cpu --gpus 0 --cpus 16 "
    "--mem 64 --time 1h --name nv2-r5-pin-cost "
    "--outputs 'results/noise_v2/rounds/round5/calibration/**' -- bash -lc '"
    "export PYTHONPATH=src; "
    "python scripts/noise_v2_supports.py build --set dregon-floor "
    "--out results/noise_v2/rounds/round5/supports; "
    "python scripts/noise_v2_calibrate_dregon.py pin --pool fit --allow-fit-pool "
    "--out results/noise_v2/rounds/round5/calibration_fitpool'"
)
#: What each priced pool IS, recorded beside every number priced on it.
COST_POOL_NOTE = {
    "score": (
        "priced on the THREE 4 s SCORE WINDOWS' own cells (the windows the pin is bisected "
        "on and the trackers are scored on), NOT on the fit's five 8 s segments: the fit "
        "pool's forward model is a cluster job, see fit_pool_cost.command"
    ),
    "fit": "priced on the fit's OWN five 8 s pooled segments, the cells it was fitted on",
}
#: Where the DREGON flight support caches live on THIS machine. The fit
#: rebuilt its own inside its cluster job and `supports.CACHE_DIR` (round 1)
#: carries bench points only, so the objective below reads R3's caches — the
#: same `.npz` files, built by the same builder from the same recordings.
SUPPORT_CACHE_DIRS = (
    Path("results/noise_v2/rounds/round5/supports"),
    Path("results/noise_v2/rounds/round3/supports"),
    Path("results/noise_v2/rounds/round2/supports"),
)

#: The comparanda of the score table are re-derivations of committed numbers,
#: read from the probe's own JSON rather than re-scored.
PROBE_TRACKS = Path("results/noise_v2/rounds/round5/tracker_probe/tracks.json")


def die(m: str) -> NoReturn:
    raise SystemExit(f"error: {m}")


def _module(name: str) -> Any:
    path = Path("scripts") / f"{name}.py"
    if not path.exists():
        die(f"{path}: not found (run from the repository root)")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
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


# ── the statistic ───────────────────────────────────────────────────────────


class Reader:
    """The probe's CQT read of one rig at one offset, cached per window.

    Every piece of this is the probe's own code (`noise_v2_tracker_probe`):
    the front end is the frozen HPPNet's own `CQTLogSpecgram` module, the
    label is sampled at the CQT's frame centres, the geometry and the ladder
    are built from the label alone, and the carrier gain is
    `alpha_curve`'s. The render is the widen runner's (`mutate` + the frozen
    seed and microphone count), so an offset of 0 reproduces the probe's
    numbers for a rig the probe already read.
    """

    def __init__(self, fit: dict[str, Any], *, recordings: tuple[str, ...]) -> None:
        self.tp = _module("noise_v2_tracker_probe")
        self.wd = _module("noise_v2_widen_dregon")
        self.fit = fit
        self.fe, self.params = self.tp.cqt_frontend()
        self.bpo = int(self.params["bins_per_octave"])
        self.hop = int(self.params["hop_length"])
        self.freqs = float(self.params["fmin_hz"]) * 2.0 ** (
            np.arange(int(self.params["n_bins"])) / self.bpo
        )
        self.windows: dict[str, dict[str, Any]] = {}
        for rec in recordings:
            mat = self.tp.window_material(rec)
            n_frames = int(np.asarray(mat["real"]).shape[-1] // self.hop) + 1
            fr = self.tp.label_on_frames(mat["rps"], n_frames, self.hop)
            self.windows[rec] = dict(
                mat=mat,
                key=mat["support"].key,
                n_frames=n_frames,
                fr=fr,
                geo=self.tp.comb_geometry(fr, freqs=self.freqs, bpo=self.bpo),
            )
        self._cache: dict[tuple[str, float], dict[str, Any]] = {}

    def render(self, recording: str, offset_db: float) -> np.ndarray:
        mat = self.windows[recording]["mat"]
        return np.asarray(
            RD.render_noise(
                self.wd.mutate(self.fit, shift_db=float(offset_db)),
                mat["rps"],
                n_mics=self.tp.N_MICS,
                seed=self.tp.SEED,
            ),
            dtype=np.float64,
        )[: self.tp.N_MICS]

    def read(self, recording: str, offset_db: float) -> dict[str, Any]:
        key = (recording, round(float(offset_db), 6))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        w = self.windows[recording]
        audio = self.render(recording, offset_db)
        lspec = self.tp.cqt_db(self.fe, audio)
        if lspec.shape[1] != w["n_frames"]:
            die(f"{recording}: CQT gave {lspec.shape[1]} frames, expected {w['n_frames']}")
        n_mics = self.tp.N_MICS
        stats = [
            self.tp.ladder_stats(self.tp.contrast_ladder(lspec[m], w["geo"])) for m in range(n_mics)
        ]
        sweeps = [
            self.tp.alpha_curve(lspec[m], w["fr"], freqs=self.freqs, bpo=self.bpo)
            for m in range(n_mics)
        ]
        out = dict(
            offset_db=float(offset_db),
            carrier_gain_s8_db=float(np.mean([s["carrier_gain_db"]["8"] for s in sweeps])),
            carrier_gain_s4_db=float(np.mean([s["carrier_gain_db"]["4"] for s in sweeps])),
            carrier_gain_s16_db=float(np.mean([s["carrier_gain_db"]["16"] for s in sweeps])),
            s8_median_db=float(np.mean([p["S8_median_db"] for p in stats])),
            argmax_alpha_s8=float(np.median([float(s["argmax_alpha"]["8"]) for s in sweeps])),
            alpha_curve_s8={
                a: float(np.mean([s["curve"][a]["8"] for s in sweeps])) for a in sweeps[0]["curve"]
            },
            ladder_db={
                str(k): float(np.nanmean([p["c_k_median_db"][str(k)] for p in stats]))
                for k in LADDER_ORDERS
            },
            frac_c1_gt_3db=float(np.mean([p["frac_c_gt_3db"]["1"] for p in stats])),
            level_dbrms=float(10.0 * np.log10(np.mean(audio**2) + 1e-30)),
        )
        self._cache[key] = out
        return out

    def worst(self, offset_db: float) -> tuple[float, dict[str, dict[str, Any]]]:
        """The carrier gain of the WEAKEST window at this offset, and the rows."""
        rows = {rec: self.read(rec, offset_db) for rec in self.windows}
        return min(r["carrier_gain_s8_db"] for r in rows.values()), rows

    def evaluated(self) -> list[dict[str, Any]]:
        out = []
        for (rec, _off), row in sorted(self._cache.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            out.append(dict(recording=rec, **row))
        return out


def bisect_offset(reader: Reader) -> dict[str, Any]:
    """The smallest grid offset whose WEAKEST window clears the threshold.

    The grid is `OFFSET_LO .. OFFSET_HI` in `OFFSET_STEP`; the bisection needs
    the pass set to be upward-closed, which the audit sweep in the payload
    verifies rather than assumes. Both ends are evaluated first, so an offset
    that already passes at 0 and one that never passes are reported as such
    instead of being bisected into a wrong answer.
    """
    grid = np.round(np.arange(OFFSET_LO, OFFSET_HI + 0.5 * OFFSET_STEP, OFFSET_STEP), 6)
    lo_gain, _ = reader.worst(float(grid[0]))
    if lo_gain >= S8_THRESHOLD:
        return dict(offset_db=float(grid[0]), needed=False, bracketed=True, n_evaluations=1)
    hi_gain, _ = reader.worst(float(grid[-1]))
    if hi_gain < S8_THRESHOLD:
        return dict(
            offset_db=None,
            needed=True,
            bracketed=False,
            top_of_search_gain_db=float(hi_gain),
            n_evaluations=2,
        )
    lo, hi = 0, int(grid.size) - 1  # grid[lo] fails, grid[hi] passes
    while hi - lo > 1:
        mid = (lo + hi) // 2
        gain, _ = reader.worst(float(grid[mid]))
        if gain >= S8_THRESHOLD:
            hi = mid
        else:
            lo = mid
    return dict(
        offset_db=float(grid[hi]),
        last_failing_db=float(grid[lo]),
        needed=True,
        bracketed=True,
        n_evaluations=len({k[1] for k in reader._cache}),
    )


# ── the price ───────────────────────────────────────────────────────────────


def score_window_specs(recordings: tuple[str, ...]) -> list[Any]:
    """The three 4 s SCORE windows, as support specs of the fit's own set.

    THE PIN IS PRICED HERE AND NOWHERE ELSE. `dregon-floor` carries both
    halves of the R5 arrangement — the five 4 s SCORED windows and the five
    disjoint 8 s FLOOR segments the fit was run on — and this returns the
    scored half, restricted to the recordings the pin was bisected on. The
    8 s fit pool is never built by this script: see `WHITTLE_FRAME_CHUNK` for
    what its forward model costs.
    """
    from experiments.noise_model import supports as SU

    want = [str(r) for r in recordings]
    found = {
        str(s.args["recording"]): s
        for s in SU.support_set(SUPPORT_SET)
        if float(s.args.get("dur_s", 0.0)) == float(SU.DREGON_SCORED_DUR_S)
    }
    missing = [r for r in want if r not in found]
    if missing:
        die(
            f"support set {SUPPORT_SET!r} carries no "
            f"{SU.DREGON_SCORED_DUR_S:g} s scored window for {missing}"
        )
    return [found[r] for r in want]


def support_cache() -> Path:
    """The directory holding the `.npz` caches of the fit's own pool.

    The first of `SUPPORT_CACHE_DIRS` that carries the DREGON flight caches.
    Rebuilding them from the raw recordings instead would cost ~20 minutes and
    300 MB for bit-identical arrays, so a missing cache is an error, not a
    silent fall-back.
    """
    for d in SUPPORT_CACHE_DIRS:
        if d.is_dir() and any(d.glob("flight_dregon_*.npz")):
            return d
    die(f"no DREGON flight support cache in {[str(d) for d in SUPPORT_CACHE_DIRS]}")


def fit_pool_specs(fit: dict[str, Any]) -> list[Any]:
    """The fit's OWN five 8 s pooled supports, by the names it records.

    Only `--pool fit --allow-fit-pool` reaches this, and only on a machine
    that is allowed to build it (see `cost_batch`).
    """
    from experiments.noise_model import supports as SU

    names = [str(s) for s in fit["supports"]]
    by_name = {s.name: s for s in SU.support_set(SUPPORT_SET)}
    missing = [n for n in names if n not in by_name]
    if missing:
        die(f"support set {SUPPORT_SET!r} carries no {missing}")
    return [by_name[n] for n in names]


def cost_batch(fit: dict[str, Any], *, pool: str, recordings: tuple[str, ...]) -> Any:
    """The cells the pin is priced on, as one flight batch on the fit's grid.

    `pool="score"` is the three 4 s SCORE windows — the windows the pin is
    bisected on and the trackers are scored on, and the ONLY pool this step
    is allowed to build on the laptop. `pool="fit"` is the fit's own five 8 s
    segments, which is a cluster job (`FIT_POOL_COST_COMMAND`).

    The front end, the frame stride and the order cap come off the fit
    payload, so either way the cells are the model's own. The fit's `k_max`
    is carried over rather than recomputed from this pool's carrier: the
    profile the objective reads has exactly that many orders.
    """
    from experiments.noise_model import model as MD
    from experiments.noise_model import supports as SU

    nvf = _module("noise_v2_fit")
    cache = support_cache()
    specs = fit_pool_specs(fit) if pool == "fit" else score_window_specs(recordings)
    loaded = [SU.load_support(str(s.text), cache_dir=cache) for s in specs]
    members = [
        (
            s.name,
            np.asarray(s.power, dtype=np.float64),
            np.asarray(s.carrier_rev_s_audio, dtype=np.float64),
            np.asarray(s.frame_starts, dtype=np.int64),
        )
        for s in loaded
    ]
    first = loaded[0]
    b = dict((fit.get("diagnostics") or {}).get("batch") or {})
    batch = MD.flight_batch(
        name=("dregon_fit_pool" if pool == "fit" else "dregon_score_windows"),
        members=members,
        sr=int(first.sr),
        n_fft=int(first.n_fft),
        hop=int(first.hop),
        k_cap=int(nvf.K_CAP),
        frame_stride=int(b.get("frame_stride", SCORE_FRAME_STRIDE)),
        max_frames=FIT_MAX_FRAMES if pool == "fit" else None,
    )
    if pool == "fit":
        got, want = int(batch.power.shape[1]), int(b["n_frames_used"])
        if got != want:
            die(f"rebuilt fit pool has {got} frames, the fit recorded {want}")
    k_fit = int(fit["k_max"])
    if int(batch.k_max) != k_fit:
        batch = replace(batch, k_max=k_fit)
    return batch


def whittle_nats_per_cell(
    batch: Any, fit: dict[str, Any], offset_db: float, *, frame_chunk: int = WHITTLE_FRAME_CHUNK
) -> dict[str, Any]:
    """The model's own Whittle objective on this batch, in FRAME CHUNKS.

    `composite_risk` is `sum_i a_i sum_{m, f in band} [I_i / M_i + log M_i]`
    — a weighted sum over frames with per-frame weights the batch already
    carries — so summing it over disjoint frame chunks with their own slice
    of `weights` (NOT `batch_slice`, which rescales the weights to the full
    set) returns the whole batch's value exactly, at a fraction of the peak
    memory.
    """
    import torch

    from experiments.noise_model import model as MD

    wd = _module("noise_v2_widen_dregon")
    par = MD.params_from_dict(wd.mutate(fit, shift_db=float(offset_db))["params"])
    n = int(batch.power.shape[1])
    step = max(1, int(frame_chunk))
    acc = dict(all=0.0, floor=0.0, comb=0.0)
    with torch.no_grad():
        for c0 in range(0, n, step):
            sl = slice(c0, min(c0 + step, n))
            sub = replace(
                batch,
                power=batch.power[:, sl],
                weights=batch.weights[sl],
                rate_work=batch.rate_work[:, sl],
            )
            m_model = MD.forward(sub, par)
            for key, band in (("all", sub.band), ("floor", sub.band_lo), ("comb", sub.band_hi)):
                acc[key] += float(MD.composite_risk(sub.power, m_model, sub.weights, band=band))
            del m_model, sub
    n_mf = int(batch.power.shape[0]) * int(batch.power.shape[1])
    cells = dict(
        floor=n_mf * int(batch.band_lo.sum()),
        comb=n_mf * int(batch.band_hi.sum()),
    )
    return dict(
        offset_db=float(offset_db),
        whittle_nats=float(acc["all"]),
        n_cells=int(batch.n_cells),
        nats_per_cell=float(acc["all"]) / float(batch.n_cells),
        per_band=dict(floor=float(acc["floor"]), comb=float(acc["comb"])),
        n_cells_per_band=cells,
        n_frames=n,
        frame_chunk=step,
        exposure_scale=float(batch.exposure_scale),
    )


def write_pinned_fit(
    fit: dict[str, Any], *, offset_db: float, calibration: dict[str, Any], path: Path
) -> Path:
    """The pinned rig AS A FIT JSON, so every frozen renderer can read it.

    `profile_db` carries the offset already applied and `comb_gain_db` records
    what it was re-levelled BY — the same contract `write_fit` states for a
    frozen-comb fit: recorded so the shift stays auditable, NEVER to be
    applied a second time. `diagnostics.calibration` says what the offset is
    and what it cost, so no reader can mistake this rig for a fitted one.
    """
    wd = _module("noise_v2_widen_dregon")
    out = wd.mutate(fit, shift_db=float(offset_db))
    out["params"]["profile"]["comb_gain_db"] = float(offset_db)
    out["diagnostics"] = dict(out.get("diagnostics") or {}) | dict(calibration=calibration)
    out["calibration_of"] = str(out.pop("_path", None) or fit.get("_path") or "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return path


# ── (1) the pin ─────────────────────────────────────────────────────────────


def run_pin(
    *,
    fit_path: Path,
    out: Path,
    recordings: tuple[str, ...],
    pool: str = "score",
    with_cost: bool = True,
) -> dict[str, Any]:
    fit = json.loads(Path(fit_path).read_text())
    fit["_path"] = str(fit_path)
    reader = Reader(fit, recordings=recordings)
    print(f"# bisecting on {len(recordings)} windows, threshold S8 gain >= {S8_THRESHOLD} dB")
    result = bisect_offset(reader)
    offset = result.get("offset_db")
    if offset is None:
        die(
            "no offset in the search range clears the threshold: "
            f"{result.get('top_of_search_gain_db'):.3f} dB at +{OFFSET_HI} dB"
        )
    # the monotonicity audit, on a coarse grid the bisection did not choose
    for a in np.round(np.arange(OFFSET_LO, OFFSET_HI + 1e-9, AUDIT_STEP), 6):
        reader.worst(float(a))
    at0 = {rec: reader.read(rec, 0.0) for rec in recordings}
    atpin = {rec: reader.read(rec, offset) for rec in recordings}

    recorded = (
        float(fit["objective"]["whittle_nats"]) / float(fit["objective"]["n_cells"])
        if fit.get("objective")
        else None
    )
    cost0: dict[str, Any] | None = None
    costpin: dict[str, Any] | None = None
    if with_cost:
        batch = cost_batch(fit, pool=pool, recordings=recordings)
        print(
            f"# pricing on the {pool} pool: {int(batch.power.shape[1])} frames, "
            f"{batch.n_cells} cells, {WHITTLE_FRAME_CHUNK} frames per forward",
            flush=True,
        )
        cost0 = whittle_nats_per_cell(batch, fit, 0.0)
        costpin = whittle_nats_per_cell(batch, fit, offset)
        del batch

    audit: dict[str, list[float]] = {}
    for rec in recordings:
        rows = sorted(
            (r for r in reader.evaluated() if r["recording"] == rec), key=lambda r: r["offset_db"]
        )
        audit[rec] = [float(r["carrier_gain_s8_db"]) for r in rows]
    offsets = sorted({float(r["offset_db"]) for r in reader.evaluated()})
    monotone = {
        rec: bool(np.all(np.diff(np.asarray(v, dtype=np.float64)) >= -1e-9))
        for rec, v in audit.items()
    }

    calibration = dict(
        offset_db=float(offset),
        s8_threshold=float(S8_THRESHOLD),
        nats_per_cell_cost=(
            None
            if costpin is None or cost0 is None
            else float(costpin["nats_per_cell"] - cost0["nats_per_cell"])
        ),
        cost_pool=(pool if with_cost else None),
        cost_note=(
            COST_POOL_NOTE[pool]
            if with_cost
            else (
                "NOT COMPUTED: this pass ran with --no-cost. The Whittle objective of the "
                "flight forward model is the one step of this study that does not fit on the "
                "laptop at the fit pool's size; the command that computes it on the cluster "
                f"is: {FIT_POOL_COST_COMMAND}"
            )
        ),
    )
    pinned_path = write_pinned_fit(
        fit, offset_db=float(offset), calibration=calibration, path=out / f"{PIN_STEM}.json"
    )

    payload: dict[str, Any] = dict(
        schema=SCHEMA,
        study="pin",
        git=git_rev(),
        fit=dict(
            path=str(fit_path),
            support=fit.get("support"),
            mode=fit.get("mode"),
            converged=(fit.get("optimiser") or {}).get("converged"),
            which_converged=(fit.get("optimiser") or {}).get("which_converged"),
        ),
        protocol=dict(
            statistic=(
                "S8 carrier gain on the frozen HPPNet's own CQT front end: the harmonic sum "
                "over k <= 8 of the read bin over the inter-tooth floor, at the label's "
                "carrier, minus its median over the mis-tuned carriers of "
                "noise_v2_tracker_probe.ALPHA_GRID. Mean over 8 microphones, per window"
            ),
            statistic_code="scripts/noise_v2_tracker_probe.py: alpha_curve / ladder_stats",
            render_code="scripts/noise_v2_widen_dregon.py: mutate + render.render_noise",
            frontend=self_params(reader),
            seed=reader.tp.SEED,
            n_mics=reader.tp.N_MICS,
            threshold_db=S8_THRESHOLD,
            threshold_provenance=(
                "round5/tracker_probe/findings.md: the ON arms sit at +5.57 (v2_legacy_free), "
                "+5.69 (legacy) and +5.98 dB (v2_legacy_lowk); the OFF arm at +1.63 (v2_real)"
            ),
            search=dict(lo_db=OFFSET_LO, hi_db=OFFSET_HI, step_db=OFFSET_STEP),
            offset_semantics=(
                "one scalar dB added to profile_db of EVERY rotor and EVERY order, i.e. the "
                "model's comb_gain_db, which sample_params folds into the profile and the "
                "renderer therefore reads already applied"
            ),
            recordings=list(recordings),
        ),
        pin=result | calibration,
        windows={
            rec: dict(
                key=reader.windows[rec]["key"],
                at_zero=at0[rec],
                at_pin=atpin[rec],
                delta_carrier_gain_db=float(
                    atpin[rec]["carrier_gain_s8_db"] - at0[rec]["carrier_gain_s8_db"]
                ),
            )
            for rec in recordings
        },
        likelihood=dict(
            pool=(pool if with_cost else None),
            priced_on=(
                list(recordings)
                if (with_cost and pool == "score")
                else (list(fit.get("supports") or []) if with_cost else [])
            ),
            note=calibration["cost_note"],
            at_zero=cost0,
            at_pin=costpin,
            nats_per_cell_cost=calibration["nats_per_cell_cost"],
            # the fit's own recorded objective, on ITS OWN five 8 s segments:
            # a different pool and a different cell count, quoted as context
            # and NOT as a comparandum of the two numbers above
            fit_pool_recorded_nats_per_cell=recorded,
            fit_pool_cost=dict(
                status="not_computed",
                reason=(
                    "the flight forward model on the fit's own 255-frame pool is the step "
                    "that OOM-killed this terminal three times; the laptop rule for R5 is "
                    "that it is never built here"
                ),
                command=FIT_POOL_COST_COMMAND,
            )
            if pool != "fit"
            else dict(status="computed", command=FIT_POOL_COST_COMMAND),
        ),
        audit=dict(
            offsets_db=offsets,
            carrier_gain_db=audit,
            monotone_in_offset=monotone,
            rule=(
                "every offset the bisection or the coarse sweep evaluated, in order: the "
                "bisection is only valid if the pass set is upward-closed"
            ),
        ),
        pinned_fit=str(pinned_path),
    )
    payload["summary"] = dict(
        offset_db=float(offset),
        carrier_gain_at_zero_db={r: at0[r]["carrier_gain_s8_db"] for r in recordings},
        carrier_gain_at_pin_db={r: atpin[r]["carrier_gain_s8_db"] for r in recordings},
        ladder_at_zero_db={
            str(k): float(np.mean([at0[r]["ladder_db"][str(k)] for r in recordings]))
            for k in LADDER_ORDERS
        },
        ladder_at_pin_db={
            str(k): float(np.mean([atpin[r]["ladder_db"][str(k)] for r in recordings]))
            for k in LADDER_ORDERS
        },
        cost_pool=(pool if with_cost else None),
        nats_per_cell_at_zero=(None if cost0 is None else cost0["nats_per_cell"]),
        nats_per_cell_at_pin=(None if costpin is None else costpin["nats_per_cell"]),
        nats_per_cell_cost=calibration["nats_per_cell_cost"],
    )
    return payload


def self_params(reader: Reader) -> dict[str, Any]:
    p = dict(reader.params)
    p.pop("scorer", None)
    return p


def run_check(
    *, fit_path: Path, out: Path, recordings: tuple[str, ...], offset_db: float
) -> dict[str, Any]:
    """Does an offset bisected on some windows also clear the gate on OTHERS?

    The same statistic, the same render route and the same threshold as
    `run_pin`, evaluated at ONE given offset instead of searched for: the
    question is whether the pin the three score windows bought carries to the
    rest of the frozen cruise cohort, not what those windows would have
    bisected to on their own.
    """
    fit = json.loads(Path(fit_path).read_text())
    fit["_path"] = str(fit_path)
    reader = Reader(fit, recordings=recordings)
    rows: dict[str, Any] = {}
    for rec in recordings:
        at0 = reader.read(rec, 0.0)
        atpin = reader.read(rec, float(offset_db))
        rows[rec] = dict(
            key=reader.windows[rec]["key"],
            at_zero=at0,
            at_pin=atpin,
            clears_at_zero=bool(at0["carrier_gain_s8_db"] >= S8_THRESHOLD),
            clears_at_pin=bool(atpin["carrier_gain_s8_db"] >= S8_THRESHOLD),
            margin_db=float(atpin["carrier_gain_s8_db"] - S8_THRESHOLD),
        )
        print(
            f"[{rec}] S8 {at0['carrier_gain_s8_db']:.3f} -> {atpin['carrier_gain_s8_db']:.3f} dB "
            f"at +{offset_db:g} dB, threshold {S8_THRESHOLD} "
            f"({'CLEARS' if rows[rec]['clears_at_pin'] else 'FAILS'})",
            flush=True,
        )
    return dict(
        schema=SCHEMA,
        study="check",
        git=git_rev(),
        fit=dict(path=str(fit_path), mode=fit.get("mode")),
        protocol=dict(
            offset_db=float(offset_db),
            threshold_db=S8_THRESHOLD,
            statistic_code="scripts/noise_v2_tracker_probe.py: alpha_curve / ladder_stats",
            render_code="scripts/noise_v2_widen_dregon.py: mutate + render.render_noise",
            seed=reader.tp.SEED,
            n_mics=reader.tp.N_MICS,
            recordings=list(recordings),
            question=(
                "the pin was bisected on the three probe windows; these are the REST of the "
                "frozen DREGON cruise cohort at that same offset"
            ),
        ),
        windows=rows,
        summary=dict(
            offset_db=float(offset_db),
            carrier_gain_at_zero_db={r: rows[r]["at_zero"]["carrier_gain_s8_db"] for r in rows},
            carrier_gain_at_pin_db={r: rows[r]["at_pin"]["carrier_gain_s8_db"] for r in rows},
            all_clear_at_pin=bool(all(rows[r]["clears_at_pin"] for r in rows)),
        ),
    )


# ── (2) the score ───────────────────────────────────────────────────────────


def _pit(payload: dict[str, Any], arm: str, key: str = "pit_mae") -> dict[str, Any]:
    rows = payload["supports"]
    per = {k: float(v["arms"][arm][key]) for k, v in rows.items()}
    return dict(per_window=per, mean=float(np.mean(list(per.values()))))


def stability_fit(pinned_path: Path, *, extra_db: float, out: Path) -> Path:
    """The pinned rig `extra_db` HIGHER, as a fit JSON: the stability arm.

    The pin is the SMALLEST offset that crosses a threshold, so the verdict it
    buys is only useful if it does not fall apart just above the crossing.
    This is the same rig with the same one scalar moved up, written the same
    way, and the calibration block says what it is.
    """
    pinned = json.loads(Path(pinned_path).read_text())
    wd = _module("noise_v2_widen_dregon")
    out_fit = wd.mutate(pinned, shift_db=float(extra_db))
    base = float((pinned.get("params") or {})["profile"].get("comb_gain_db") or 0.0)
    cal = dict((pinned.get("diagnostics") or {}).get("calibration") or {})
    out_fit["params"]["profile"]["comb_gain_db"] = base + float(extra_db)
    out_fit["diagnostics"] = dict(out_fit.get("diagnostics") or {}) | dict(
        calibration=cal
        | dict(
            offset_db=base + float(extra_db),
            stability_margin_db=float(extra_db),
            role=(
                "STABILITY ARM, not the pin: the pinned rig with the same scalar "
                f"{extra_db:g} dB higher, scored to show the verdict is not balanced on "
                "the threshold crossing"
            ),
        )
    )
    path = Path(out) / f"{PIN_STEM}_plus{int(round(extra_db))}db.json"
    path.write_text(json.dumps(out_fit, indent=1, sort_keys=True) + "\n")
    return path


def run_score(
    *,
    fit_path: Path,
    pinned_path: Path,
    out: Path,
    recordings: tuple[str, ...],
    trackers: tuple[str, ...],
    stability_db: float = 3.0,
) -> dict[str, Any]:
    """HPPNet and SCv2 on the three windows: as-is, +pin, +pin+`stability_db`.

    The rendering and scoring are the widen runner's `run`, called as a
    library with no change: same three windows, same seed 2001, same eight
    microphones, same `revised_eval` PIT path. The `real` and `legacy` arms
    come along on the as-is pass as the identity check — they must reproduce
    the probe's committed numbers, since the audio is bit-identical.
    """
    wd = _module("noise_v2_widen_dregon")
    tp = _module("noise_v2_tracker_probe")
    plus_path = stability_fit(pinned_path, extra_db=stability_db, out=out)
    scored: dict[str, Any] = {}
    for tracker in trackers:
        spec = tp.TRACKERS[tracker]
        kw = (
            dict(probe_experiment=None, probe_ckpt="best")
            if spec is None
            else dict(probe_experiment=spec["experiment"], probe_ckpt=spec["ckpt"])
        )
        print(f"# {tracker}: fit as-is (real, legacy, v2)", flush=True)
        asis = wd.run(
            fit_path=fit_path,
            out=out,
            probe=True,
            recordings=recordings,
            only=("real", "legacy", "v2"),
            **kw,
        )
        asis.pop("_figures", None)
        print(f"# {tracker}: fit + pin (v2)", flush=True)
        pinned = wd.run(
            fit_path=pinned_path, out=out, probe=True, recordings=recordings, only=("v2",), **kw
        )
        pinned.pop("_figures", None)
        print(f"# {tracker}: fit + pin + {stability_db:g} dB (v2)", flush=True)
        plus = wd.run(
            fit_path=plus_path, out=out, probe=True, recordings=recordings, only=("v2",), **kw
        )
        plus.pop("_figures", None)
        scored[tracker] = dict(
            scorer=asis["protocol"]["scorer"],
            real=_pit(asis, "real"),
            legacy=_pit(asis, "legacy"),
            v2_r5_asis=_pit(asis, "v2"),
            v2_r5_pin=_pit(pinned, "v2"),
            v2_r5_pin_plus=_pit(plus, "v2"),
            widths={
                name: {
                    k: v["arms"]["v2"]["widths"][0]["width_hz"]
                    for k, v in payload["supports"].items()
                }
                for name, payload in (
                    ("v2_r5_asis", asis),
                    ("v2_r5_pin", pinned),
                    ("v2_r5_pin_plus", plus),
                )
            },
        )
    quoted = quoted_arms()
    return dict(
        schema=SCHEMA,
        study="score",
        git=git_rev(),
        protocol=dict(
            runner="scripts/noise_v2_widen_dregon.py: run (unchanged, called as a library)",
            recordings=list(recordings),
            seed=wd.SEED,
            n_mics=8,
            trackers={k: tp.TRACKERS[k] for k in trackers},
            fits=dict(
                asis=str(fit_path),
                pin=str(pinned_path),
                pin_plus=str(plus_path),
                stability_db=float(stability_db),
            ),
            quoted_from=str(PROBE_TRACKS),
        ),
        scored=scored,
        quoted=quoted,
        table=score_table(scored, quoted),
    )


def quoted_arms() -> dict[str, Any]:
    """The R4/R3 comparanda, read off the probe's committed JSON."""
    if not PROBE_TRACKS.exists():
        return {}
    d = json.loads(PROBE_TRACKS.read_text())
    out: dict[str, Any] = {}
    for arm in ("real", "legacy", "v2_real", "v2_legacy_lowk", "v2_legacy_free"):
        row = {}
        for tracker in d["summary"]:
            s = d["summary"][tracker].get(arm)
            if s is not None:
                row[tracker] = float(s["pit_mae"])
        if row:
            out[arm] = row
    return out


def score_table(scored: dict[str, Any], quoted: dict[str, Any]) -> list[dict[str, Any]]:
    """Every arm of the verdict in one table, the quoted ones first.

    The quoted rows are NOT re-scored here: they are the probe's committed
    numbers on the same three windows, the same seed and the same two
    trackers, read off `tracks.json`.
    """
    rows: list[dict[str, Any]] = []
    for arm, label in (
        ("real", "the real DREGON room-2 clip"),
        ("legacy", "legacy stage-2 render"),
        ("v2_real", "R3 v2 fitted to the real clip (flight_floor_lowk)"),
        ("v2_legacy_lowk", "R4 v2 fitted to the LEGACY render (flight_floor_lowk)"),
        ("v2_legacy_free", "R4 v2 fitted to the LEGACY render (flight, free profile)"),
    ):
        rows.append(
            dict(
                arm=arm,
                label=label,
                source="round5/tracker_probe/tracks.json",
                **quoted.get(arm, {}),
            )
        )
    for arm, label in (
        ("v2_r5_asis", "R5 v2 flight_profile, as fitted"),
        ("v2_r5_pin", "R5 v2 flight_profile + the calibration pin"),
        ("v2_r5_pin_plus", "R5 v2 flight_profile + the pin + the stability margin"),
    ):
        rows.append(
            dict(
                arm=arm,
                label=label,
                source="this pass",
                **{t: scored[t][arm]["mean"] for t in scored},
            )
        )
    return rows


# ── (3) figures ─────────────────────────────────────────────────────────────


def probe_carrier_gains() -> list[tuple[str, float, str]]:
    """The committed S8 carrier gain of the probe's own arms, for the figure.

    Read from `round5/tracker_probe/cqt.json`, never retyped: the REAL clip's
    own gain (0.98 dB, tracked at 1.07 rev/s) is the strongest evidence that
    the 5.6 dB threshold is a SUFFICIENT side of a boundary and not a
    necessary condition, and it belongs on the same axes as the pin.
    """
    path = PROBE_TRACKS.parent / "cqt.json"
    if not path.is_file():
        return []
    summary = json.loads(path.read_text())["summary"]
    want = (("real", "tab:green"), ("legacy", "tab:brown"), ("v2_real", "tab:red"))
    out: list[tuple[str, float, str]] = []
    for arm, colour in want:
        row = (summary.get(arm) or {}).get("carrier_gain_db")
        # the probe records the gain per harmonic-sum depth; S8 is the statistic
        if isinstance(row, dict) and "8" in row:
            out.append((arm, float(row["8"]), colour))
    return out


def write_figures(pin: dict[str, Any], score: dict[str, Any] | None, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[str] = []
    recs = list(pin["protocol"]["recordings"])
    off = np.asarray(pin["audit"]["offsets_db"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    for rec in recs:
        ax.plot(
            off, pin["audit"]["carrier_gain_db"][rec], marker="o", ms=3, label=rec.split("_")[0]
        )
    ax.axhline(S8_THRESHOLD, color="k", ls="--", lw=1, label=f"ON threshold {S8_THRESHOLD} dB")
    ax.axvline(
        pin["pin"]["offset_db"], color="crimson", lw=1, label=f"pin +{pin['pin']['offset_db']:g} dB"
    )
    for name, val, c in probe_carrier_gains():
        ax.axhline(val, color=c, lw=0.8, alpha=0.6)
        ax.text(
            off[0],
            val + 0.25,
            f"{name} {val:.2f} dB",
            va="bottom",
            ha="left",
            fontsize=7,
            color=c,
        )
    ax.set_xlim(off[0] - 0.8, off[-1] + 0.8)
    ax.set_xlabel("comb offset applied to profile_db (dB)")
    ax.set_ylabel("S8 carrier gain on HPPNet's CQT (dB)")
    ax.set_title("R5 DREGON: the calibration pin is a threshold crossing, not a fit")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    p = out / "carrier_gain_vs_offset.png"
    fig.savefig(p)
    plt.close(fig)
    written.append(str(p))

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    ks = [int(k) for k in LADDER_ORDERS]
    ax.plot(
        ks, [pin["summary"]["ladder_at_zero_db"][str(k)] for k in ks], marker="o", label="as fitted"
    )
    ax.plot(
        ks,
        [pin["summary"]["ladder_at_pin_db"][str(k)] for k in ks],
        marker="s",
        label=f"+{pin['pin']['offset_db']:g} dB pin",
    )
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("order k")
    ax.set_ylabel("c_k: read bin over the inter-tooth floor (dB)")
    ax.set_title("R5 DREGON: the CQT prominence ladder, as fitted and pinned")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "prominence_ladder.png"
    fig.savefig(p)
    plt.close(fig)
    written.append(str(p))

    if score is not None:
        fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
        rows = score["table"]
        names = [r["arm"] for r in rows]
        x = np.arange(len(rows), dtype=np.float64)
        for i, tracker in enumerate(("hppnet", "scv2")):
            vals = [float(r.get(tracker, np.nan)) for r in rows]
            ax.bar(x + 0.35 * (i - 0.5), vals, width=0.33, label=tracker)
        ax.axhline(2.187786, color="crimson", ls="--", lw=1, label="DREGON parity bar 2.19")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("3-window PIT MAE (rev/s, log)")
        ax.set_title(f"R5 DREGON: both trackers, {len(rows)} arms, three cruise score windows")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out / "score_arms.png"
        fig.savefig(p)
        plt.close(fig)
        written.append(str(p))
    return written


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pin", "check", "score", "figures"):
        p = sub.add_parser(name)
        p.add_argument("--fit", type=Path, default=FIT_DEFAULT)
        p.add_argument("--out", type=Path, default=OUT_DEFAULT)
        p.add_argument("--recordings", default=None)
        if name == "pin":
            p.add_argument(
                "--pool",
                choices=sorted(COST_POOL_NOTE),
                default="score",
                help="cells the pin is priced on (default: the three 4 s score windows)",
            )
            p.add_argument(
                "--allow-fit-pool",
                action="store_true",
                help="permit --pool fit; a CLUSTER switch, see FIT_POOL_COST_COMMAND",
            )
            p.add_argument(
                "--no-cost",
                action="store_true",
                help="skip the price entirely; nats_per_cell_cost is then null with a reason",
            )
        if name == "score":
            p.add_argument("--pinned-fit", type=Path, default=None)
            p.add_argument("--trackers", default="hppnet,scv2")
        if name == "check":
            p.add_argument(
                "--offset-db",
                type=float,
                default=None,
                help="the offset to test (default: the pin recorded in <out>/pin.json)",
            )
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tp = _module("noise_v2_tracker_probe")
    recordings = tuple(
        r for r in (args.recordings or ",".join(tp.RECORDINGS)).split(",") if r.strip()
    )

    if args.cmd == "pin":
        if args.pool == "fit" and not args.allow_fit_pool:
            die(
                "--pool fit builds the flight forward model on the fit's own five 8 s "
                "segments; that is a cluster job (it OOM-killed this laptop three times). "
                f"Pass --allow-fit-pool to force it, or run: {FIT_POOL_COST_COMMAND}"
            )
        payload = run_pin(
            fit_path=args.fit,
            out=out,
            recordings=recordings,
            pool=str(args.pool),
            with_cost=not args.no_cost,
        )
        (out / "pin.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(json.dumps(payload["summary"], indent=1))
        print(f"wrote {out / 'pin.json'} and {payload['pinned_fit']}")
        return 0

    if args.cmd == "check":
        offset = args.offset_db
        if offset is None:
            pin_json = out / "pin.json"
            if not pin_json.is_file():
                die(f"no --offset-db and no {pin_json} to read the pin from")
            offset = float(json.loads(pin_json.read_text())["pin"]["offset_db"])
        payload = run_check(
            fit_path=args.fit, out=out, recordings=recordings, offset_db=float(offset)
        )
        (out / "check.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(json.dumps(payload["summary"], indent=1))
        print(f"wrote {out / 'check.json'}")
        return 0

    if args.cmd == "score":
        pinned = args.pinned_fit or (out / f"{PIN_STEM}.json")
        payload = run_score(
            fit_path=args.fit,
            pinned_path=Path(pinned),
            out=out,
            recordings=recordings,
            trackers=tuple(t for t in str(args.trackers).split(",") if t),
        )
        (out / "score.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(json.dumps(payload["table"], indent=1))
        print(f"wrote {out / 'score.json'}")
        return 0

    pin = json.loads((out / "pin.json").read_text())
    sc = out / "score.json"
    written = write_figures(pin, json.loads(sc.read_text()) if sc.exists() else None, out)
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
