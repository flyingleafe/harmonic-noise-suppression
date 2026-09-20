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
fit's own five-window support, and the difference in nats per cell is what the
pin costs in likelihood. Nothing here is a fit and nothing here is free: one
scalar, one threshold, one price.

    # bisect the offset, price it, write the pinned rig as a fit JSON
    python scripts/noise_v2_calibrate_dregon.py pin \
        --fit results/noise_v2/rounds/round5/fits/dregon_room2_floor__flight_profile.json
    # score HPPNet + SCv2 on the three windows, as-is and pinned
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
#: `noise_v2_fit.py flight --max-frames`, the value every DREGON flight fit of
#: this campaign ran with. The fit payload records the frames it USED, not the
#: cap that produced them, so the rebuilt batch is checked against it.
FIT_MAX_FRAMES = 256
#: Where the five pooled DREGON flight support caches live on THIS machine.
#: The fit rebuilt them inside its own job and `supports.CACHE_DIR` (round 1)
#: carries bench points only, so the objective below reads R3's caches — the
#: same `.npz` files, built by the same builder from the same recordings, and
#: the `n_frames_used` check in `fit_batch` fails loudly if they are not.
SUPPORT_CACHE_DIRS = (
    Path("results/noise_v2/rounds/round5/supports"),
    Path("results/noise_v2/rounds/round3/supports"),
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


def fit_specs(fit: dict[str, Any], *, support_set: str = "dregon-floor") -> list[str]:
    """The support SPECS behind the names a fit payload records.

    A fit lists its pool by support NAME; `load_support` takes a spec. The
    named set is the same one the fit was launched with, so resolving through
    it — and failing loudly on a name the set does not carry — keeps the
    objective below on exactly the fit's own five windows.
    """
    from experiments.noise_model import supports as SU

    names = [str(s) for s in fit["supports"]]
    by_name = {s.name: s.text for s in SU.support_set(support_set)}
    missing = [n for n in names if n not in by_name]
    if missing:
        die(f"support set {support_set!r} carries no {missing}")
    return [by_name[n] for n in names]


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


def fit_batch(fit: dict[str, Any]) -> Any:
    """The fit's OWN support batch, rebuilt from the fit JSON's own record.

    The five pooled supports, the frame stride and the frame cap all come off
    the fit payload, so the objective below is evaluated on exactly the cells
    the fit was scored on and the fitted rig reproduces its own recorded
    `objective.whittle_nats`.
    """
    from experiments.noise_model import model as MD
    from experiments.noise_model import supports as SU

    nvf = _module("noise_v2_fit")
    cache = support_cache()
    specs = fit_specs(fit)
    loaded = [SU.load_support(str(spec), cache_dir=cache) for spec in specs]
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
    b = dict(fit["diagnostics"]["batch"])
    batch = MD.flight_batch(
        name=str(fit["support"]),
        members=members,
        sr=int(first.sr),
        n_fft=int(first.n_fft),
        hop=int(first.hop),
        k_cap=int(nvf.K_CAP),
        frame_stride=int(b["frame_stride"]),
        max_frames=FIT_MAX_FRAMES,
    )
    got = int(np.asarray(batch.power.shape)[1])
    if got != int(b["n_frames_used"]):
        die(
            f"rebuilt batch has {got} frames, the fit recorded {b['n_frames_used']}: "
            "the objective below would not be on the fit's own cells"
        )
    return batch


def whittle_nats_per_cell(batch: Any, fit: dict[str, Any], offset_db: float) -> dict[str, Any]:
    """The fit's own Whittle objective with the comb offset applied."""
    import torch

    from experiments.noise_model import model as MD

    wd = _module("noise_v2_widen_dregon")
    par = MD.params_from_dict(wd.mutate(fit, shift_db=float(offset_db))["params"])
    with torch.no_grad():
        obj = MD.objective_breakdown(batch, MD.forward(batch, par))
    return dict(
        offset_db=float(offset_db),
        whittle_nats=float(obj["whittle_nats"]),
        n_cells=int(obj["n_cells"]),
        nats_per_cell=float(obj["whittle_nats"]) / float(obj["n_cells"]),
        per_band={k: float(v) for k, v in obj["per_band"].items()},
        n_cells_per_band={k: int(v) for k, v in obj["n_cells_per_band"].items()},
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


def run_pin(*, fit_path: Path, out: Path, recordings: tuple[str, ...]) -> dict[str, Any]:
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
    batch = fit_batch(fit)
    cost0 = whittle_nats_per_cell(batch, fit, 0.0)
    costpin = whittle_nats_per_cell(batch, fit, offset)

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
        nats_per_cell_cost=float(costpin["nats_per_cell"] - cost0["nats_per_cell"]),
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
            support=fit.get("support"),
            n_supports=len(fit.get("supports") or []),
            at_zero=cost0,
            at_pin=costpin,
            nats_per_cell_cost=calibration["nats_per_cell_cost"],
            recorded_in_fit=recorded,
            # the batch rebuild is only trustworthy if the UNSHIFTED rig
            # reproduces the objective the fit itself recorded
            reproduces_recorded_rel=(
                None
                if recorded is None
                else abs(cost0["nats_per_cell"] - recorded) / max(abs(recorded), 1e-12)
            ),
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
        nats_per_cell_at_zero=cost0["nats_per_cell"],
        nats_per_cell_at_pin=costpin["nats_per_cell"],
        nats_per_cell_cost=calibration["nats_per_cell_cost"],
    )
    return payload


def self_params(reader: Reader) -> dict[str, Any]:
    p = dict(reader.params)
    p.pop("scorer", None)
    return p


# ── (2) the score ───────────────────────────────────────────────────────────


def _pit(payload: dict[str, Any], arm: str, key: str = "pit_mae") -> dict[str, Any]:
    rows = payload["supports"]
    per = {k: float(v["arms"][arm][key]) for k, v in rows.items()}
    return dict(per_window=per, mean=float(np.mean(list(per.values()))))


def run_score(
    *,
    fit_path: Path,
    pinned_path: Path,
    out: Path,
    recordings: tuple[str, ...],
    trackers: tuple[str, ...],
) -> dict[str, Any]:
    """HPPNet and SCv2 on the three windows, fit-as-is and fit+pin.

    The rendering and scoring are the widen runner's `run`, called as a
    library with no change: same three windows, same seed 2001, same eight
    microphones, same `revised_eval` PIT path. The `real` and `legacy` arms
    come along on the as-is pass as the identity check — they must reproduce
    the probe's committed numbers, since the audio is bit-identical.
    """
    wd = _module("noise_v2_widen_dregon")
    tp = _module("noise_v2_tracker_probe")
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
            fit_path=pinned_path,
            out=out,
            probe=True,
            recordings=recordings,
            only=("v2",),
            **kw,
        )
        pinned.pop("_figures", None)
        scored[tracker] = dict(
            scorer=asis["protocol"]["scorer"],
            real=_pit(asis, "real"),
            legacy=_pit(asis, "legacy"),
            v2_r5_asis=_pit(asis, "v2"),
            v2_r5_pin=_pit(pinned, "v2"),
            widths=dict(
                v2_r5_asis={
                    k: v["arms"]["v2"]["widths"][0]["width_hz"] for k, v in asis["supports"].items()
                },
                v2_r5_pin={
                    k: v["arms"]["v2"]["widths"][0]["width_hz"]
                    for k, v in pinned["supports"].items()
                },
            ),
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
            fits=dict(asis=str(fit_path), pin=str(pinned_path)),
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
    rows: list[dict[str, Any]] = []
    for arm, label in (
        ("real", "the real DREGON room-2 clip"),
        ("legacy", "legacy stage-2 render"),
        ("v2_real", "R3 v2 fitted to the real clip (flight_floor_lowk)"),
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
    for name, val, c in (
        ("real", 0.983, "tab:green"),
        ("legacy", 5.687, "tab:brown"),
    ):
        ax.axhline(val, color=c, lw=0.8, alpha=0.6)
        ax.text(off[-1], val, f" {name}", va="center", fontsize=7, color=c)
    ax.set_xlabel("comb offset applied to profile_db (dB)")
    ax.set_ylabel("S8 carrier gain on HPPNet's CQT (dB)")
    ax.set_title("R5 DREGON: the calibration pin is a threshold crossing, not a fit")
    ax.legend(fontsize=7)
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
        ax.set_title("R5 DREGON: both trackers, five arms")
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
    for name in ("pin", "score", "figures"):
        p = sub.add_parser(name)
        p.add_argument("--fit", type=Path, default=FIT_DEFAULT)
        p.add_argument("--out", type=Path, default=OUT_DEFAULT)
        p.add_argument("--recordings", default=None)
        if name == "score":
            p.add_argument("--pinned-fit", type=Path, default=None)
            p.add_argument("--trackers", default="hppnet,scv2")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tp = _module("noise_v2_tracker_probe")
    recordings = tuple(
        r for r in (args.recordings or ",".join(tp.RECORDINGS)).split(",") if r.strip()
    )

    if args.cmd == "pin":
        payload = run_pin(fit_path=args.fit, out=out, recordings=recordings)
        (out / "pin.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(json.dumps(payload["summary"], indent=1))
        print(f"wrote {out / 'pin.json'} and {payload['pinned_fit']}")
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
