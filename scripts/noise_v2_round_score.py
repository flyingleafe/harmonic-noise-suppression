"""Score one noise-model-v2 round against the three FROZEN gates.

One command per round. It renders the round's candidate on the REAL carriers of
the frozen supports, runs the frozen HPPNet probe, and writes the round record
plus its findings:

    python scripts/noise_v2_round_score.py --round 1 --fits results/noise_v2/rounds/round1/fits
    python scripts/noise_v2_round_score.py --round 1 --legacy-export baseline --out-tag legacy_smoke

The measurement protocol is the previous campaign's, reproduced from the code
that produced the recorded numbers; it is written down in
``experiments.noise_model.gates`` and not restated here. What this script owns:

* **the arm.** ``--fits DIR`` reads the round's ``noise-v2-fit/1`` JSONs and
  renders through ``experiments.noise_model.render.render_noise``. Each rig's
  arm is ONE fit, selected by its ``support`` name: DREGON renders from
  ``dregon_room2_floor`` (the flight floor-only fit, whose comb block is the
  mean of the four ``bench_dregon_Motor{1-4}_70`` bench fits — read as well and
  verified, not assumed), Michael's from ``michaels_fly125_cruise``, which
  drives all three FLY124 regimes (standby and ramp are not fitted — approved
  decision 3 — and are rendered on their own real carrier through the
  renderer's trajectory sampler). ``--legacy-export baseline`` instead renders
  the OLD model — the legacy stage-2 exports the frozen evaluator selected per
  cohort/regime — so the whole pipeline can be smoke tested against the
  recorded current-best numbers (DREGON cruise synthetic PIT MAE
  ``2.1877858830655468``; Michael's equal-regime mean ``3.026661398168452``)
  before the v2 renderer exists.
* **the frozen render seeds.** Every arm is rendered at
  ``null_variation.seeds = [2001, 2002, 2003, 2004]`` and each support's PIT MAE
  is the mean over those seeds, which is the evaluator's ``per_window``
  quantity. The proxy gate reads the FIRST seed's render (the criteria study
  used one render seed) and the per-seed spread is recorded next to it.
* **the likelihood cells.** For Michael's two FLY124 cruise supports it builds
  the observed periodogram at NFFT 2048 / hop 512, the arm's expected
  periodogram on the same grid, and the speed-matched stationary oracle — the
  Welch mean periodogram of the disjoint reference segment recorded in
  ``results/noise_v2/short_whittle/short_whittle.json``, recomputed here with
  ``noise_v2_likelihood_window.oracle_mean_periodogram`` so the per-band split
  exists.

Writes ``results/noise_v2/rounds/round<N>.json`` and
``results/noise_v2/rounds/round<N>/score/findings.md``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.noise_model import gates as GT
from experiments.stochastic_fit import revised_eval as RE
from experiments.stochastic_fit import stage2 as S2
from experiments.stochastic_fit.data import Clip, periodogram

OUT_DEFAULT = Path("results/noise_v2/rounds")
ORACLE_JSON = Path("results/noise_v2/short_whittle/short_whittle.json")
FIT_SCHEMA = "noise-v2-fit/1"

#: The fit each rig's arm is RENDERED from, by the ``"support"`` field of the
#: fit JSON (R1Core's naming). The DREGON arm is the flight floor-only fit
#: whose comb block is already the mean of the four single-rotor bench fits;
#: those four are read too, purely to verify that mean.
DREGON_FIT_SUPPORT = "dregon_room2_floor"
DREGON_COMB_SUPPORTS: tuple[str, ...] = tuple(f"bench_dregon_Motor{i}_70" for i in (1, 2, 3, 4))
MICHAELS_FIT_SUPPORT = "michaels_fly125_cruise"
#: The comb parameters the DREGON arm carries over from the bench.
COMB_PARAM_KEYS: tuple[str, ...] = (
    "sigma_nu",
    "lam",
    "sigma_eps_even",
    "sigma_eps_odd",
    "lam_eps_even",
    "lam_eps_odd",
)

#: The legacy stage-2 export the frozen evaluator selected per (rig, regime),
#: with the route ``baseline_params`` reads it through
#: (``scripts/stochastic_fit_revised_eval.py:639-692``). ``identity`` uses that
#: recording's own fitted clips; ``aggregate`` is the declared cross-recording
#: extrapolation Michael's held-out FLY124 needs.
LEGACY_BASELINE: dict[tuple[str, str], dict[str, Any]] = {
    ("dregon", "cruise"): dict(
        path="results/S2/dregon_room2_cruise_refined.json",
        family="refined",
        regime="cruise",
        match="identity",
        declared=None,
    ),
    ("michaels", "cruise"): dict(
        path="results/S2/cruise_8clip_refined.json",
        family="refined",
        regime="cruise",
        match="aggregate",
        declared=None,
    ),
    ("michaels", "ramp"): dict(
        path="results/S2/cruise_8clip_refined.json",
        family="refined",
        regime="cruise",
        match="aggregate",
        declared=None,
    ),
    ("michaels", "standby"): dict(
        path="results/S2/standby.json",
        family="raw",
        regime="standby",
        match="aggregate",
        declared=dict(
            recordings=["FLY125"],
            starts_s=[1.78],
            seconds=13.5,
            dataset="michaels-frames",
            version=None,
            rps_key="rps",
        ),
    ),
}

#: The OTHER eligible family of a rig's cruise export. Which family the frozen
#: evaluator actually selected is recorded only in the gitignored
#: ``results/revised_phase/baseline_v2/calibration.json``, so the smoke must be
#: able to state and to vary it: ``--legacy-family michaels=raw``. On DREGON
#: the raw export (``results/S2/dregon_flight.json``) names one recording only
#: and the frozen manifest reports it INELIGIBLE for the five-recording cohort,
#: so there is nothing to vary there.
LEGACY_FAMILY_ALTERNATIVES: dict[tuple[str, str], dict[str, Any]] = {
    ("michaels", "raw"): dict(
        path="results/S2/cruise_8clip.json",
        family="raw",
        regime="cruise",
        match="aggregate",
        declared=dict(
            recordings=["FLY125"],
            starts_s=[16.0, 32.0, 48.0, 64.0, 96.0, 112.0, 128.0, 144.0],
            seconds=16.0,
            dataset="michaels-frames",
            version=None,
            rps_key="rps",
        ),
    ),
}


def die(message: str) -> None:
    raise SystemExit(f"error: {message}")


def _module(name: str) -> Any:
    """One script of this tree, imported by path (the study's own idiom)."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        die(f"cannot load {path}")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


# ── the arms ────────────────────────────────────────────────────────────────


@dataclass
class Arm:
    """The round's candidate on one rig: render and expected periodogram."""

    rig: str
    kind: str  # v2 | legacy
    label: str
    source: dict[str, Any]
    fit: dict[str, Any] | None = None
    legacy: dict[str, RE.ModelParams] | None = None  # per regime
    _render: Any = None
    _spectrum: Any = None

    def render(self, rps: np.ndarray, *, regime: str, n_mics: int, seed: int) -> np.ndarray:
        if self.kind == "v2":
            assert self.fit is not None and self._render is not None
            audio = self._render(self.fit, np.atleast_2d(rps), n_mics=int(n_mics), seed=int(seed))
            return np.asarray(audio, dtype=np.float64)
        assert self.legacy is not None
        mp = self.legacy[regime]
        physical = RE.to_renderer_units(mp)
        return np.asarray(
            S2.render_from_export(
                dict(physical.params),
                np.atleast_2d(rps),
                sample_rate_work=RE.SAMPLE_RATE_WORK,
                n_mics=int(n_mics),
                seed=int(seed),
                normalize_rms=None,
            ),
            dtype=np.float64,
        )

    def spectrum(
        self, clip: Clip, *, regime: str, n_mics: int, n_fft: int, hop: int
    ) -> np.ndarray | None:
        """``(M, N, F)`` expected periodogram on the given front end."""
        if self.kind == "v2":
            if self._spectrum is None:
                return None
            assert self.fit is not None
            m = self._spectrum(
                self.fit,
                np.asarray(clip.rps, dtype=np.float64),
                n_fft=int(n_fft),
                hop=int(hop),
                sr=int(clip.sr),
                n_mics=int(n_mics),
            )
            return np.asarray(m, dtype=np.float64)[:n_mics]
        assert self.legacy is not None
        pg = periodogram(clip, n_fft=int(n_fft), hop=int(hop))
        return np.asarray(RE.predicted_m(self.legacy[regime], pg, n_mics=int(n_mics)), np.float64)


def legacy_arm(rig: str, *, family: str | None = None) -> Arm:
    """The OLD model's arm: the frozen evaluator's own baseline route.

    ``family`` overrides the CRUISE-derived entries with the rig's other
    eligible export family (see :data:`LEGACY_FAMILY_ALTERNATIVES`); the
    standby entry has only one export and is never overridden.
    """
    per_regime: dict[str, RE.ModelParams] = {}
    source: dict[str, Any] = {}
    alt = LEGACY_FAMILY_ALTERNATIVES.get((rig, str(family))) if family else None
    if family and alt is None:
        die(f"no alternative {family!r} export is registered for rig {rig!r}")
    for (arm_rig, regime), base_cfg in LEGACY_BASELINE.items():
        if arm_rig != rig:
            continue
        cfg = dict(base_cfg)
        if alt is not None and str(base_cfg["regime"]) == "cruise":
            cfg = dict(alt)
            cfg["overrides_family"] = base_cfg["family"]
        bundle = RE.read_export(
            cfg["path"], family=cfg["family"], regime=cfg["regime"], declared=cfg["declared"]
        )
        if cfg["match"] == "identity":
            per_regime[regime] = _identity_placeholder(bundle, regime)
        else:
            per_regime[regime] = RE.aggregate_nuisance(
                bundle,
                list(bundle.clip_ids),
                label=f"{cfg['family']}:{cfg['regime']}:aggregate",
                extrapolated=True,
            )
        source[regime] = dict(cfg, clip_ids=list(bundle.clip_ids))
    if not per_regime:
        die(f"no legacy baseline export is registered for rig {rig!r}")
    return Arm(
        rig=rig,
        kind="legacy",
        label=(
            f"legacy stage-2 baseline, {family} family on the cruise export"
            if family
            else "legacy stage-2 baseline (the frozen evaluator's selected family)"
        ),
        source=source,
        legacy=per_regime,
    )


_IDENTITY_BUNDLES: dict[str, Any] = {}


def _identity_placeholder(bundle: Any, regime: str) -> RE.ModelParams:
    """``match: identity`` resolves per recording, so keep the bundle around."""
    _IDENTITY_BUNDLES[regime] = bundle
    return RE.aggregate_nuisance(
        bundle, list(bundle.clip_ids), label=f"identity-pending:{regime}", extrapolated=True
    )


def legacy_identity_params(arm: Arm, *, regime: str, recording: str) -> RE.ModelParams:
    """The identity-matched legacy parameter set of one recording."""
    cfg = arm.source[regime]
    if cfg["match"] != "identity":
        if arm.legacy is None:
            die(f"arm {arm.rig} has no legacy parameter set")
        assert arm.legacy is not None
        return arm.legacy[regime]
    bundle = _IDENTITY_BUNDLES[regime]
    pairs = bundle.clips_of(recording)
    if not pairs:
        die(
            f"{cfg['path']} does not name {recording!r} (it names {list(bundle.recordings)}); "
            "identity pairing has no positional fallback"
        )
    return RE.aggregate_nuisance(
        bundle,
        [cid for cid, _ in pairs],
        label=f"{cfg['family']}:{cfg['regime']}:identity:{recording}",
    )


def read_fits(fits_dir: Path) -> dict[str, dict[str, Any]]:
    """Every ``noise-v2-fit/1`` JSON of a round, indexed by its ``support``."""
    files = sorted(Path(fits_dir).glob("*.json"))
    if not files:
        die(f"{fits_dir}: no fit JSON found — the round's fits must be committed before scoring")
    out: dict[str, dict[str, Any]] = {}
    for p in files:
        d = json.loads(p.read_text())
        if str(d.get("schema")) != FIT_SCHEMA:
            continue
        support = str(d.get("support") or "")
        if not support:
            die(f"{p}: a {FIT_SCHEMA} fit must name its support")
        d["_path"] = str(p)
        out[support] = d
    if not out:
        die(f"{fits_dir}: no file carries schema {FIT_SCHEMA!r}")
    return out


def v2_arm(rig: str, fits_dir: Path, *, render_mod: Any, spectrum_fn: Any) -> Arm:
    """The round's v2 arm: one fit per rig, selected by its ``support`` name.

    DREGON renders from the flight floor-only fit ``dregon_room2_floor``, whose
    comb block is the mean of the four ``bench_dregon_Motor{1-4}_70`` bench
    fits; those four are read as well and the mean is VERIFIED rather than
    assumed. Michael's renders from ``michaels_fly125_cruise`` for all three
    FLY124 regimes.
    """
    fits = read_fits(fits_dir)
    want = MICHAELS_FIT_SUPPORT if rig == "michaels" else DREGON_FIT_SUPPORT
    if want not in fits:
        die(
            f"{fits_dir}: no fit carries support {want!r} (found {sorted(fits)}); the {rig} arm "
            "is not defined by any other file"
        )
    fit = fits[want]
    source: dict[str, Any] = dict(support=want, path=fit["_path"], kind=fit.get("kind"))
    label = (
        "v2 FLY125 cruise fit (drives FLY124 standby/ramp through the trajectory sampler)"
        if rig == "michaels"
        else "v2 DREGON room-2 floor fit with the bench comb (mean of Motor1-4 @70)"
    )
    if rig == "dregon":
        source["comb_mean_check"] = _comb_mean_check(fit, fits)
    return Arm(
        rig=rig,
        kind="v2",
        label=label,
        source=source,
        fit=fit,
        _render=render_mod.render_noise,
        _spectrum=spectrum_fn,
    )


def _comb_mean_check(fit: dict[str, Any], fits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Is the rendered DREGON comb really the mean of the four bench fits?"""
    missing = [s for s in DREGON_COMB_SUPPORTS if s not in fits]
    if missing:
        return dict(available=False, missing_bench_fits=missing)
    params = [fits[s]["params"] for s in DREGON_COMB_SUPPORTS]
    rows: dict[str, Any] = {}
    for key in COMB_PARAM_KEYS:
        if key not in fit["params"] or any(key not in p for p in params):
            continue
        mean = float(np.mean([float(p[key]) for p in params]))
        got = float(fit["params"][key])
        rows[key] = dict(
            bench_mean=mean,
            rendered=got,
            relative=(abs(got - mean) / abs(mean) if mean else None),
        )
    rel = [r["relative"] for r in rows.values() if r["relative"] is not None]
    return dict(
        available=bool(rows),
        bench_fits=list(DREGON_COMB_SUPPORTS),
        per_parameter=rows,
        max_relative=(max(rel) if rel else None),
        rule="the floor-only flight fit must carry the bench comb mean unchanged",
    )


# ── measurement ─────────────────────────────────────────────────────────────


@dataclass
class Probe:
    """The frozen tracker, loaded once and pinned by digest."""

    record: dict[str, Any]
    tracker: Any

    @classmethod
    def load(cls) -> Probe:
        ev = _module("stochastic_fit_revised_eval")
        record = ev.checkpoint_record(GT.SCORER_EXPERIMENT, GT.SCORER_CKPT)
        if record["sha256"] != GT.SCORER_SHA256:
            die(
                f"frozen scorer digest mismatch: {record['sha256']} != {GT.SCORER_SHA256}; "
                "the probe checkpoint moved and nothing may be scored against it"
            )
        return cls(
            record=record,
            tracker=ev.Tracker(dict(experiment=GT.SCORER_EXPERIMENT, ckpt=GT.SCORER_CKPT)),
        )


def measure_support(
    support: GT.ScoredSupport,
    *,
    arm: Arm,
    probe: Probe | None,
    seeds: tuple[int, ...],
    n_mics: int,
) -> dict[str, Any]:
    """Real and candidate measurements of one frozen support."""
    clip = RE.load_window(
        support.window,
        dataset=GT.DATASET[support.rig],
        version=None,
        channels=None,
        rps_key=GT.RAW_RPS_KEY[support.rig],
    )
    real_all = np.asarray(clip.audio, dtype=np.float64)
    mics = list(range(min(int(n_mics), int(real_all.shape[0]))))
    real = real_all[: len(mics)]
    reference = np.atleast_2d(np.asarray(clip.rps, dtype=np.float64))
    n_samples = int(real.shape[-1])
    rsupport = RE.regime_support(
        support.window,
        reference,
        regime=support.regime,
        min_rps=support.min_rps,
        max_rps=support.max_rps,
        sr=clip.sr,
    )
    if rsupport.n_scored == 0:
        die(f"{support.key}: the frozen {support.regime} band selects no sample of this support")
    row: dict[str, Any] = dict(
        support=support.as_dict(),
        n_samples=n_samples,
        mics=mics,
        scored_seconds=rsupport.scored_seconds,
        scored_fraction=float(rsupport.n_scored / max(rsupport.sample_mask.size, 1)),
        mean_reference_rps=float(reference.mean()),
        per_rotor_mean_rps=[float(v) for v in reference.mean(axis=1)],
        seeds=list(seeds),
        per_seed=[],
    )
    if probe is not None:
        real_pit = probe.tracker.pit(
            real, reference, mics=mics, expected_samples=n_samples, support=rsupport
        )
        row["real_pit_mae"] = float(real_pit["mae"])
        row["real_pit"] = real_pit
    renders: dict[int, np.ndarray] = {}
    if arm.kind == "legacy" and arm.legacy is not None:
        arm = _arm_for_recording(arm, support)
    for seed in seeds:
        t0 = time.time()
        audio = arm.render(reference, regime=support.regime, n_mics=len(mics), seed=int(seed))[
            : len(mics)
        ]
        if int(audio.shape[-1]) != n_samples:
            die(
                f"{support.key}: the arm rendered {audio.shape[-1]} samples against the frozen "
                f"{n_samples}; refusing to score it on another timeline"
            )
        renders[int(seed)] = audio
        entry: dict[str, Any] = dict(seed=int(seed), render_seconds=time.time() - t0)
        if probe is not None:
            pit = probe.tracker.pit(
                audio, reference, mics=mics, expected_samples=n_samples, support=rsupport
            )
            entry["pit_mae"] = float(pit["mae"])
            entry["pit_per_mic"] = pit["per_mic"]
            entry["pit_per_rotor"] = pit["per_rotor"]
            entry["n_scored_frames"] = pit["n_scored_frames"]
        entry["ltas_abs_db"] = float(RE.ltas_deviation_db(real, audio, mic=0)["mean_abs_db"])
        row["per_seed"].append(entry)
    if probe is not None:
        maes = [e["pit_mae"] for e in row["per_seed"]]
        row["pit_mae"] = float(np.mean(maes))
        row["pit_mae_seed_spread"] = float(max(maes) - min(maes))
    ltas = [e["ltas_abs_db"] for e in row["per_seed"]]
    row["ltas_abs_db_first_seed"] = float(row["per_seed"][0]["ltas_abs_db"])
    row["ltas_abs_db_seed_mean"] = float(np.mean(ltas))
    row["ltas_abs_db_seed_spread"] = float(max(ltas) - min(ltas))
    return row | dict(_clip=clip, _real=real, _render=renders[int(seeds[0])], _arm=arm)


def _arm_for_recording(arm: Arm, support: GT.ScoredSupport) -> Arm:
    """Resolve an identity-matched legacy arm onto this support's recording."""
    legacy = dict(arm.legacy or {})
    legacy[support.regime] = legacy_identity_params(
        arm, regime=support.regime, recording=support.recording
    )
    return Arm(
        rig=arm.rig,
        kind="legacy",
        label=arm.label,
        source=arm.source,
        legacy=legacy,
    )


# ── the likelihood cells ────────────────────────────────────────────────────


def oracle_reference(oracle_json: dict[str, Any], support: GT.ScoredSupport) -> dict[str, Any]:
    for entry in oracle_json.get("supports") or []:
        if str((entry.get("scored") or {}).get("key")) == support.key:
            ref = entry.get("oracle_reference")
            if not ref:
                die(f"{support.key}: short_whittle.json records no oracle reference segment")
            return dict(ref, oracle_match=entry.get("oracle_match"))
    die(f"{support.key}: not present in {ORACLE_JSON}")
    raise AssertionError


def likelihood_cell(
    support: GT.ScoredSupport,
    row: dict[str, Any],
    *,
    arm: Arm,
    oracle_json: dict[str, Any],
    lw: Any,
    n_fft: int,
    hop: int,
    n_mics: int,
) -> tuple[GT.SpectralCell | None, dict[str, Any]]:
    clip: Clip = row["_clip"]
    mics = min(int(n_mics), int(np.asarray(clip.audio).shape[0]))
    pg = periodogram(clip, n_fft=int(n_fft), hop=int(hop))
    power = np.asarray(pg.power, dtype=np.float64)[:mics]
    model = arm.spectrum(clip, regime=support.regime, n_mics=mics, n_fft=n_fft, hop=hop)
    ref = oracle_reference(oracle_json, support)
    ref_support = lw.Support(
        support.rig,
        str(ref["recording"]),
        float(ref["start_s"]),
        float(ref["duration_s"]),
        str(ref.get("regime", support.regime)),
        "oracle_reference",
        str(ref.get("rule", "recorded in short_whittle.json")),
    )
    ref_clip = lw.load(ref_support, rps_key=GT.RAW_RPS_KEY[support.rig], channels=None)
    oracle = np.broadcast_to(
        lw.oracle_mean_periodogram(ref_clip, n_fft=int(n_fft), hop=int(hop), n_mics=mics),
        power.shape,
    )
    detail = dict(
        support=support.key,
        n_frames=int(power.shape[1]),
        n_mics=mics,
        oracle_reference=ref,
        model_available=model is not None,
    )
    if model is None:
        detail["unavailable"] = (
            "the arm exposes no expected periodogram on this front end, so the likelihood gate "
            "cannot be evaluated"
        )
        return None, detail
    if model.shape != power.shape:
        die(
            f"{support.key}: the arm's expected periodogram {model.shape} does not match the "
            f"frozen observation {power.shape}"
        )
    cell = GT.SpectralCell(
        support=support,
        freqs_hz=np.asarray(pg.freqs, dtype=np.float64),
        frame_times_s=np.asarray(pg.times, dtype=np.float64),
        power=power,
        model=np.asarray(model, dtype=np.float64),
        oracle=np.asarray(oracle, dtype=np.float64),
    )
    return cell, detail


# ── the round ───────────────────────────────────────────────────────────────


def run(
    *,
    round_no: int,
    fits_dir: Path | None,
    legacy: bool,
    legacy_families: dict[str, str],
    seeds: tuple[int, ...],
    n_mics: int,
    rigs: tuple[str, ...],
    with_probe: bool,
) -> dict[str, Any]:
    lw = _module("noise_v2_likelihood_window")
    proxy_mod = _module("noise_v2_spectrogram_proxy")
    oracle_json = json.loads(ORACLE_JSON.read_text()) if ORACLE_JSON.is_file() else None
    arms: dict[str, Arm] = {}
    for rig in rigs:
        if legacy:
            arms[rig] = legacy_arm(rig, family=legacy_families.get(rig))
        else:
            render_mod, spectrum_fn = _v2_modules()
            assert fits_dir is not None
            arms[rig] = v2_arm(rig, fits_dir, render_mod=render_mod, spectrum_fn=spectrum_fn)
    probe = Probe.load() if with_probe else None
    payload: dict[str, Any] = dict(
        round=int(round_no),
        git=git_rev(),
        arm=dict(
            kind="legacy" if legacy else "v2",
            label={r: a.label for r, a in arms.items()},
            source={r: a.source for r, a in arms.items()},
        ),
        fits=(
            sorted(str(p) for p in Path(fits_dir).glob("*.json"))
            if fits_dir is not None and not legacy
            else []
        ),
        protocol=dict(
            render_seeds=list(seeds),
            n_mics=int(n_mics),
            scorer=(probe.record if probe is not None else None),
            proxy_seed=int(seeds[0]),
            likelihood_front_end=dict(n_fft=GT.LIKELIHOOD_N_FFT, hop=GT.LIKELIHOOD_HOP),
            note=(
                "PIT MAE per support is the mean over the frozen render seeds (the evaluator's "
                "per_window quantity); the proxy reads the first seed's render with the per-seed "
                "spread recorded beside it"
            ),
        ),
        measurements={},
        likelihood_cells=[],
    )
    supports = []
    if "dregon" in rigs:
        supports += list(GT.DREGON_CRUISE_SUPPORTS)
    if "michaels" in rigs:
        supports += list(GT.MICHAELS_SUPPORTS)
    rows: dict[str, dict[str, Any]] = {}
    real_audio: dict[str, np.ndarray] = {}
    synth_audio: dict[str, np.ndarray] = {}
    cells: list[GT.SpectralCell] = []
    for support in supports:
        row = measure_support(
            support, arm=arms[support.rig], probe=probe, seeds=seeds, n_mics=n_mics
        )
        real_audio[support.key] = row["_real"]
        synth_audio[support.key] = row["_render"]
        if support.rig == "michaels" and support.regime == "cruise" and oracle_json is not None:
            cell, detail = likelihood_cell(
                support,
                row,
                arm=row["_arm"],
                oracle_json=oracle_json,
                lw=lw,
                n_fft=GT.LIKELIHOOD_N_FFT,
                hop=GT.LIKELIHOOD_HOP,
                n_mics=n_mics,
            )
            payload["likelihood_cells"].append(detail)
            if cell is not None:
                cells.append(cell)
        rows[support.key] = {k: v for k, v in row.items() if not k.startswith("_")}
        print(
            f"[{support.rig} {support.key}] "
            f"pit={row.get('pit_mae')} real={row.get('real_pit_mae')} "
            f"ltas={row['ltas_abs_db_first_seed']:.4f} dB",
            flush=True,
        )
    payload["measurements"] = rows
    pred = {k: v["pit_mae"] for k, v in rows.items() if "pit_mae" in v}
    real = {k: v["real_pit_mae"] for k, v in rows.items() if "real_pit_mae" in v}
    gates: dict[str, Any] = {}
    gates["hppnet"] = (
        GT.hppnet_gate(pred, real)
        if pred
        else dict(name="hppnet_pit_mae", unavailable="no probe run", **{"pass": False})
    )
    gates["proxy"] = GT.proxy_gate(
        real_audio,
        synth_audio,
        mr_ltas_fn=lambda r, a: float(
            proxy_mod.cand_mr_ltas(r, a, w_freqs=None, w=None, floor_db=None)["value"]
        ),
    )
    gates["likelihood"] = (
        GT.likelihood_gate(cells, oracle_json=oracle_json)
        if cells
        else dict(
            name="likelihood_composite_risk",
            unavailable="no Michael's cruise likelihood cell was built",
            **{"pass": False},
        )
    )
    payload["gates"] = gates
    payload["pass"] = bool(all(bool(g.get("pass")) for g in gates.values()))
    return payload


def _v2_modules() -> tuple[Any, Any]:
    """``experiments.noise_model.render`` plus its expected-periodogram hook.

    Imported dynamically: this script must run (and smoke the legacy arm)
    before the v2 renderer exists, and it must say so loudly rather than fail
    at import time.
    """
    try:
        render_mod = importlib.import_module("experiments.noise_model.render")
    except Exception as exc:
        die(f"experiments.noise_model.render is not importable yet: {exc}")
    spectrum_fn = None
    for module in (render_mod, _optional_module("experiments.noise_model.spectrum")):
        if module is None:
            continue
        for name in ("expected_periodogram", "predict_spectrum"):
            fn = getattr(module, name, None)
            if fn is not None:
                spectrum_fn = fn
                break
        if spectrum_fn is not None:
            break
    return render_mod, spectrum_fn


def _optional_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


# ── the findings ────────────────────────────────────────────────────────────


def findings(payload: dict[str, Any]) -> str:
    out: list[str] = []
    g = payload["gates"]
    verdict = "PASS" if payload["pass"] else "FAIL"
    out.append(f"# Noise model v2 — round {payload['round']} gate score ({verdict})")
    out.append("")
    out.append(
        f"Arm: {payload['arm']['kind']} — "
        + "; ".join(f"{r}: {lbl}" for r, lbl in payload["arm"]["label"].items())
    )
    out.append(
        f"Render seeds {payload['protocol']['render_seeds']}, "
        f"{payload['protocol']['n_mics']} mics, git `{payload['git']}`."
    )
    out.append("")
    out.append("## Gate verdicts")
    out.append("")
    out.append("| gate | verdict | number | threshold |")
    out.append("|---|---|---|---|")
    hp = g["hppnet"]
    if "dregon_cruise" in hp:
        d, m = hp["dregon_cruise"], hp["michaels_fly124"]
        out.append(
            f"| HPPNet DREGON cruise | {'PASS' if d['pass'] else 'FAIL'} | "
            f"{d['mean_candidate_rev_s']:.6f} rev/s (95 % upper "
            f"{_fmt(d['candidate_interval']['upper'])}) | <= {d['target_rev_s']:.6f} |"
        )
        out.append(
            f"| HPPNet Michael's ratio | {'PASS' if m['pass'] else 'FAIL'} | "
            f"{m['aggregate_ratio']:.6f} ({m['rig_candidate_mae']:.6f} rev/s) | "
            f"<= {m['ratio_max']:.2f} ({m['point_bound_rev_s']:.6f} rev/s) |"
        )
    for group, blk in g["proxy"]["groups"].items():
        out.append(
            f"| proxy `ltas_abs_db` {group} | {'PASS' if blk['pass'] else 'FAIL'} | "
            f"{blk['mean_ltas_abs_db']:.4f} dB (spread {blk['spread_ltas_abs_db']:.4f}) | "
            f"<= {blk['gate_db']:.4f} dB |"
        )
    lk = g["likelihood"]
    if "per_band" in lk:
        for band in ("comb", "floor", "full"):
            blk = lk["per_band"][band]
            decisive = band == lk["decisive_band"]
            verdict = ("PASS" if blk["below_oracle"] else "FAIL") if decisive else "report"
            out.append(
                f"| likelihood {band}{' (decisive)' if decisive else ''} | {verdict} | "
                f"model {blk['model_nats_per_s']:,.4f} nats/s | "
                f"oracle {blk['oracle_nats_per_s']:,.4f}, margin "
                f"{blk['margin_nats_per_s']:+,.4f} |"
            )
    else:
        out.append(f"| likelihood | FAIL | {lk.get('unavailable')} | — |")
    out.append("")
    out.append("## HPPNet PIT MAE per support (rev/s)")
    out.append("")
    if "dregon_cruise" not in hp:
        out.append(
            f"The HPPNet probe did not run in this pass ({hp.get('unavailable')}), so no PIT "
            "MAE is reported and the round cannot pass."
        )
        out.append("")
    out.append("| rig | support | regime | real | candidate | seed spread |")
    out.append("|---|---|---|---:|---:|---:|")
    for key, row in payload["measurements"].items():
        s = row["support"]
        out.append(
            f"| {s['rig']} | `{key}` | {s['regime']} | {_fmt(row.get('real_pit_mae'))} | "
            f"{_fmt(row.get('pit_mae'))} | {_fmt(row.get('pit_mae_seed_spread'))} |"
        )
    out.append("")
    if "dregon_cruise" in hp:
        d = hp["dregon_cruise"]
        out.append(
            f"DREGON cruise: candidate mean {d['mean_candidate_rev_s']:.6f} rev/s over "
            f"{len(d['clusters'])} recording-level clusters, one-sided 95 % interval "
            f"[{_fmt(d['candidate_interval']['lower'])}, "
            f"{_fmt(d['candidate_interval']['upper'])}] "
            f"(t and 20000-draw cluster bootstrap, conservative), against the frozen target "
            f"{d['target_rev_s']:.6f} = {d['frozen_real_rev_s']:.6f} + "
            f"{d['gap_fraction']} x ({d['frozen_baseline_rev_s']:.6f} - "
            f"{d['frozen_real_rev_s']:.6f}). Margin {_fmt(d['margin_rev_s'])} rev/s."
        )
        out.append("")
        out.append(
            f"Real arm reproduction: measured {_fmt(d['real_arm']['mean_rev_s'])} rev/s against "
            f"the frozen {d['frozen_real_rev_s']:.6f} "
            f"(relative {_fmt(d['real_arm']['relative'])})."
        )
        out.append("")
        m = hp["michaels_fly124"]
        out.append("Michael's FLY124 per regime (rev/s):")
        out.append("")
        out.append("| regime | blocks | candidate | frozen baseline | ratio |")
        out.append("|---|---:|---:|---:|---:|")
        for regime in ("standby", "ramp", "cruise"):
            r = m["per_regime"][regime]
            out.append(
                f"| {regime} | {r['n_blocks']} | {r['candidate_mae']:.6f} | "
                f"{r['baseline_mae']:.6f} | {r['ratio']:.4f} |"
            )
        out.append("")
        out.append(
            f"Equal-regime mean {m['rig_candidate_mae']:.6f} rev/s against the frozen baseline "
            f"{m['rig_baseline_mae']:.6f}; ratio {m['aggregate_ratio']:.6f} against the "
            f"{m['ratio_max']} bound ({m['arithmetic']})."
        )
    out.append("")
    out.append("## Proxy `ltas_abs_db` per support (dB, mic 0, absolute level)")
    out.append("")
    out.append("| group | support | ltas_abs_db | seed spread | mr_ltas | level offset |")
    out.append("|---|---|---:|---:|---:|---:|")
    for group, blk in g["proxy"]["groups"].items():
        for row in blk["supports"]:
            seed_spread = payload["measurements"][row["support"]]["ltas_abs_db_seed_spread"]
            out.append(
                f"| {group} | `{row['support']}` | {row['ltas_abs_db']:.4f} | "
                f"{seed_spread:.4f} | {_fmt(row['mr_ltas'], '.4f')} | "
                f"{row['level_offset_db']:+.4f} |"
            )
    out.append("")
    for group, blk in g["proxy"]["groups"].items():
        ref = blk["reference"]
        out.append(
            f"{group}: mean {blk['mean_ltas_abs_db']:.4f} dB (spread "
            f"{blk['spread_ltas_abs_db']:.4f} over {blk['n_supports']} supports) against the "
            f"closure-{g['proxy']['closure']} gate {blk['gate_db']:.4f} dB = "
            f"{ref['c3_db']:.4f} - {g['proxy']['closure']} x ({ref['c3_db']:.4f} - "
            f"{ref['oracle_db']:.4f}); margin {_fmt(blk['margin_db'], '+.4f')} dB. Secondary "
            f"`mr_ltas` {_fmt(blk['mean_mr_ltas'], '.4f')} dB (reported, never decisive)."
        )
        out.append("")
    out.append("## Likelihood: composite risk against the speed-matched stationary oracle")
    out.append("")
    if "per_band" in lk:
        out.append(
            f"NFFT {lk['front_end']['n_fft']} / hop {lk['front_end']['hop']}, pooled over "
            f"{len(lk['pooled_over'])} Michael's FLY124 cruise supports, band split at "
            f"{lk['band_edge_hz']:.0f} Hz."
        )
        out.append("")
        out.append("| band | Hz | model (nats/s) | oracle (nats/s) | margin | below oracle |")
        out.append("|---|---|---:|---:|---:|---|")
        for band in ("comb", "floor", "full"):
            blk = lk["per_band"][band]
            out.append(
                f"| {band} | {blk['band_hz'][0]:.0f}-{blk['band_hz'][1]:.0f} | "
                f"{blk['model_nats_per_s']:,.4f} | {blk['oracle_nats_per_s']:,.4f} | "
                f"{blk['margin_nats_per_s']:+,.4f} | {blk['below_oracle']} |"
            )
        out.append("")
        out.append(
            f"Frozen comb-band margin: {lk['frozen_margin_nats_per_s']:+,.4f} nats/s "
            f"({lk['frozen_margin_note']})."
        )
        rep = lk["oracle"]["reproduction"]
        if rep.get("available"):
            full = lk["per_band"]["full"]["oracle_nats_per_s"]
            rec = rep["recorded_pooled_nats_per_s"]
            out.append("")
            out.append(
                f"Oracle reproduction: recomputed full-band pooled oracle {full:,.4f} nats/s "
                f"against the recorded {rec:,.4f} nats/s of `{ORACLE_JSON}` "
                f"(relative {abs(full - rec) / abs(rec):.3e})."
            )
        else:
            out.append("")
            out.append(f"Oracle reproduction unavailable: {rep.get('reason')}")
    else:
        out.append(f"Unavailable: {lk.get('unavailable')}")
        for detail in payload.get("likelihood_cells") or []:
            if detail.get("unavailable"):
                out.append(f"* `{detail['support']}`: {detail['unavailable']}")
    out.append("")
    out.append("## Protocol provenance")
    out.append("")
    out.append(
        "Reproduced from the previous campaign (sources in "
        "`src/experiments/noise_model/gates.py`): the frozen scorer "
        "`hppnet_l2_r2_s0/best` with its SHA-256 verified before anything is scored; "
        "`revised_eval.pit_mae` over all eight microphones on the pre-registered "
        "raw-telemetry regime support; the five DREGON room-2 4 s cruise windows and the five "
        "FLY124 windows the frozen evaluator resolved; the four frozen render seeds "
        "`[2001, 2002, 2003, 2004]`; `per_window`/`per_recording` aggregation; the DREGON "
        "target as `real + 0.7 x (baseline - real)` on the frozen v2 scalars; the "
        "recording-level one-sided 95 % interval of `revised_eval.cluster_interval` "
        "(t bound and 20000-draw cluster bootstrap at seed 0, conservative side); Michael's "
        "ratio of equally weighted per-regime MAEs; `ltas_abs_db` as "
        "`revised_eval.ltas_deviation_db` mean_abs_db on mic 0 with no normalisation; the "
        "composite risk of `marginal_frame_nll` + `FrameScore` + `composite_score`; and the "
        "`oracle_np` definition and reference segments of "
        "`results/noise_v2/short_whittle/short_whittle.json`."
    )
    out.append("")
    out.append(
        "Re-derived, because the source is not in the repository or does not exist yet: "
        "(a) the DREGON paired `baseline - candidate` improvement interval needs the "
        "per-recording baseline MAEs of the gitignored "
        "`results/revised_phase/baseline_v2/calibration.json`, so the one-sided 95 % "
        "recording-level instrument is applied to the candidate's own cluster mean against the "
        "frozen scalar target instead; (b) the per-band (300 Hz) oracle risk, because "
        "`short_whittle.json` records the oracle over the full 30-7900 Hz band only — it is "
        "recomputed here from the same oracle periodogram and the full-band value is "
        "cross-checked against the recorded number; (c) v2 has no fitted standby/ramp model, so "
        "the FLY125 cruise fit drives those two regimes on their own real carriers through the "
        "renderer's trajectory sampler (approved decision 3), where the old campaign used the "
        "manifest `baseline_map` cross-recording extrapolation of the FLY125 exports."
    )
    out.append("")
    return "\n".join(out)


def _fmt(value: Any, spec: str = ".6f") -> str:
    if value is None:
        return "—"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):  # pragma: no cover
        return str(value)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="score one noise-model-v2 round against the gates")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--fits", type=Path, default=None, help="directory of noise-v2-fit/1 JSONs")
    ap.add_argument(
        "--legacy-export",
        choices=["baseline"],
        default=None,
        help="smoke the pipeline on the OLD model (the frozen evaluator's baseline exports)",
    )
    ap.add_argument(
        "--legacy-family",
        action="append",
        default=[],
        metavar="RIG=FAMILY",
        help=(
            "with --legacy-export: read that rig's cruise export from another eligible family "
            "(e.g. michaels=raw); the frozen selection itself lives in the gitignored "
            "calibration.json"
        ),
    )
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument(
        "--out-tag",
        default=None,
        help="write round<N>_<tag>.json instead of round<N>.json (smoke runs)",
    )
    ap.add_argument("--seeds", default=",".join(str(s) for s in GT.RENDER_SEEDS))
    ap.add_argument("--mics", type=int, default=8)
    ap.add_argument("--rigs", default="dregon,michaels")
    ap.add_argument(
        "--no-probe",
        action="store_true",
        help="skip the HPPNet probe (proxy and likelihood only); the round then cannot pass",
    )
    args = ap.parse_args(argv)
    legacy = args.legacy_export is not None
    if not legacy and args.fits is None:
        die("pass --fits <dir> with the round's fit JSONs, or --legacy-export baseline")
    payload = run(
        round_no=int(args.round),
        fits_dir=args.fits,
        legacy=legacy,
        legacy_families=dict(
            str(spec).split("=", 1)
            for spec in args.legacy_family  # type: ignore[misc]
        ),
        seeds=tuple(int(s) for s in str(args.seeds).replace(",", " ").split()),
        n_mics=int(args.mics),
        rigs=tuple(str(args.rigs).replace(",", " ").split()),
        with_probe=not args.no_probe,
    )
    tag = f"_{args.out_tag}" if args.out_tag else ""
    out_json = Path(args.out) / f"round{payload['round']}{tag}.json"
    out_dir = Path(args.out) / f"round{payload['round']}{tag}" / "score"
    RE.write_json(out_json, payload)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "findings.md").write_text(findings(payload))
    print(f"wrote {out_json} and {out_dir / 'findings.md'}")
    print(f"round {payload['round']}: {'PASS' if payload['pass'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
