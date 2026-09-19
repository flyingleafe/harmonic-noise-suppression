"""Score one noise-model-v2 round against the three FROZEN gates.

One command per round. It renders the round's candidate on the REAL carriers of
the frozen supports, runs the frozen HPPNet probe, and writes the round record
plus its findings:

    python scripts/noise_v2_round_score.py --round 1 --fits results/noise_v2/rounds/round1/fits
    python scripts/noise_v2_round_score.py --round 1 --legacy-export baseline --out-tag legacy_smoke

One rig at a time (one remote job each, because the DREGON and Michael's fits
land at different times), then compose the round record from the scored arms:

    python scripts/noise_v2_round_score.py --round 1 --fits <fits> --rigs michaels \
        --candidate michaels_v2_fly125cruise --arm-out <dir>/arm_michaels.json \
        --dump-audio <dir>/audio --job <omnirun job> --audio-uri s3://...
    python scripts/noise_v2_round_score.py --round 1 --compose <dir>/arm_*.json \
        --supports-index results/noise_v2/rounds/round1/supports/index.json

``--compose`` re-evaluates the joint frozen HPPNet gate on the union of the
arms' per-support PIT MAEs, carries each rig's proxy group and the Michael's
likelihood gate, and reports every HPPNet number against BOTH bars: LEGACY
PARITY (the previous best, which is the top-level pass) and the frozen
0.70-gap DREGON STRETCH target. The candidate table states each fit's own
``optimiser.converged`` status next to the numbers it produced.

The measurement protocol is the previous campaign's, reproduced from the code
that produced the recorded numbers; it is written down in
``experiments.noise_model.gates`` and not restated here. What this script owns:

* **the arm.** ``--fits DIR`` reads the round's noise-model fit JSONs (schema
  ``noise-v2-fit/2``, and R1/R2's ``/1``) and
  renders through ``experiments.noise_model.render.render_noise``. Each rig's
  arm is ONE fit, selected by its ``support`` name: DREGON renders from
  ``dregon_room2_floor`` (the flight floor-only fit, whose comb block is the
  mean of the four ``bench_dregon_Motor{1-4}_70`` bench fits — read as well and
  verified, not assumed), Michael's from one of the FLY125 flight pools
  (``michaels_fly125_all``, R2's pooled standby + ramp + cruise fit, or
  ``michaels_fly125_cruise``, R1's cruise-only pool), which drives all three
  FLY124 regimes through the renderer's trajectory sampler on each support's
  own real carrier. ``--legacy-export baseline`` instead renders
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

from experiments.noise_model import READABLE_FIT_SCHEMAS
from experiments.noise_model import gates as GT
from experiments.stochastic_fit import revised_eval as RE
from experiments.stochastic_fit import stage2 as S2
from experiments.stochastic_fit.data import Clip, periodogram

OUT_DEFAULT = Path("results/noise_v2/rounds")
ORACLE_JSON = Path("results/noise_v2/short_whittle/short_whittle.json")
#: The fit schemas an arm may be rendered from — the renderer's own pair
#: (``experiments.noise_model.render.READABLE_SCHEMAS``): this round's ``/2``
#: and R1/R2's ``/1``, whose per-order OU the renderer maps onto a width. The
#: renderer is imported dynamically (see ``_v2_modules``), so the pair is taken
#: from the package, which stays importable with no v2 renderer present.
FIT_SCHEMAS: tuple[str, ...] = READABLE_FIT_SCHEMAS

#: The fit each rig's arm is RENDERED from, by the ``"support"`` field of the
#: fit JSON (R1Core's naming). The DREGON arm is the flight floor-only fit
#: whose comb block is already the mean of the four single-rotor bench fits;
#: those four are read too, purely to verify that mean.
DREGON_FIT_SUPPORT = "dregon_room2_floor"
DREGON_COMB_SUPPORTS: tuple[str, ...] = tuple(f"bench_dregon_Motor{i}_70" for i in (1, 2, 3, 4))
#: The Michael's arm is defined on a FLY125 flight pool, and there are two
#: named pools: R2's ``michaels_fly125_all`` (8 cruise + 1 standby + 1 ramp
#: window, so both speed exponents are fitted over a 4.68x speed span) and
#: R1's cruise-only ``michaels_fly125_cruise`` (1.44x, both exponents
#: extrapolated onto standby and ramp). Directory selection takes the FIRST
#: of these that the fits directory carries; ``--fit michaels=PATH`` may name
#: either.
MICHAELS_FIT_SUPPORTS: tuple[str, ...] = ("michaels_fly125_all", "michaels_fly125_cruise")
#: The STANDBY-only fit of the per-regime Michael's candidate, named by
#: ``--fit michaels_standby=PATH``. There is no DREGON counterpart: DREGON is
#: scored on cruise windows only.
MICHAELS_STANDBY_FIT_SUPPORTS: tuple[str, ...] = ("michaels_fly125_standby",)
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
    #: the per-REGIME v2 composition: one fit per regime of
    #: ``render.REGIME_ORDER``, rendered through ``render_noise_regimes``
    regime_fits: dict[str, dict[str, Any]] | None = None
    _render: Any = None
    _spectrum: Any = None
    _render_regimes: Any = None
    _spectrum_regimes: Any = None

    def render(self, rps: np.ndarray, *, regime: str, n_mics: int, seed: int) -> np.ndarray:
        if self.kind == "v2":
            if self.regime_fits is not None:
                assert self._render_regimes is not None
                audio = self._render_regimes(
                    self.regime_fits, np.atleast_2d(rps), n_mics=int(n_mics), seed=int(seed)
                )
                return np.asarray(audio, dtype=np.float64)
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
            if self.regime_fits is not None:
                if self._spectrum_regimes is None:
                    return None
                m = self._spectrum_regimes(
                    self.regime_fits,
                    np.asarray(clip.rps, dtype=np.float64),
                    n_fft=int(n_fft),
                    hop=int(hop),
                    sr=int(clip.sr),
                    n_mics=int(n_mics),
                )
                return np.asarray(m, dtype=np.float64)[:n_mics]
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
    """Every readable fit JSON of a round, indexed by its ``support``.

    Both schemas of :data:`FIT_SCHEMAS` are taken. Where one support carries a
    fit under each, the CURRENT schema wins: a round that re-fitted a support
    under the new parameterisation is scored on the new fit.
    """
    files = sorted(Path(fits_dir).glob("*.json"))
    if not files:
        die(f"{fits_dir}: no fit JSON found — the round's fits must be committed before scoring")
    out: dict[str, dict[str, Any]] = {}
    for p in files:
        d = json.loads(p.read_text())
        schema = str(d.get("schema"))
        if schema not in FIT_SCHEMAS:
            continue
        support = str(d.get("support") or "")
        if not support:
            die(f"{p}: a {schema} fit must name its support")
        prev = out.get(support)
        if prev is not None and FIT_SCHEMAS.index(str(prev["schema"])) < FIT_SCHEMAS.index(schema):
            continue
        d["_path"] = str(p)
        out[support] = d
    if not out:
        die(f"{fits_dir}: no file carries any of the fit schemas {list(FIT_SCHEMAS)}")
    return out


def v2_arm(
    rig: str,
    fits_dir: Path,
    *,
    render_mod: Any,
    spectrum_fn: Any,
    fit_path: Path | None = None,
) -> Arm:
    """The round's v2 arm: one fit per rig, selected by its ``support`` name.

    DREGON renders from the flight floor-only fit ``dregon_room2_floor``, whose
    comb block is the mean of the four ``bench_dregon_Motor{1-4}_70`` bench
    fits; those four are read as well and the mean is VERIFIED rather than
    assumed. Michael's renders from a FLY125 flight pool — R2's pooled
    ``michaels_fly125_all`` if the directory carries it, else R1's cruise-only
    ``michaels_fly125_cruise`` — for all three FLY124 regimes.

    ``fit_path`` names ONE file instead, which is what a second candidate on
    the same support needs: a retry fit carries the same ``support`` name as
    the fit it retries, so selection by support name alone cannot tell the two
    apart. The named file must still carry one of that rig's supports.
    """
    fits = read_fits(fits_dir)
    admissible = MICHAELS_FIT_SUPPORTS if rig == "michaels" else (DREGON_FIT_SUPPORT,)
    named = " or ".join(repr(name) for name in admissible)
    if fit_path is not None:
        fit = json.loads(Path(fit_path).read_text())
        if str(fit.get("schema")) not in FIT_SCHEMAS:
            die(f"{fit_path}: schema {fit.get('schema')!r} is none of {list(FIT_SCHEMAS)}")
        if str(fit.get("support")) not in admissible:
            die(
                f"{fit_path}: carries support {fit.get('support')!r}, but the {rig} arm is "
                f"defined on {named}"
            )
        want = str(fit.get("support"))
        fit["_path"] = str(fit_path)
    else:
        present = [name for name in admissible if name in fits]
        if not present:
            die(
                f"{fits_dir}: no fit carries support {named} (found {sorted(fits)}); the {rig} "
                "arm is not defined by any other file"
            )
        want = present[0]
        fit = fits[want]
    source: dict[str, Any] = dict(support=want, path=fit["_path"], kind=fit.get("kind"))
    if rig != "michaels":
        label = "v2 DREGON room-2 floor fit with the bench comb (mean of Motor1-4 @70)"
    elif want == "michaels_fly125_all":
        label = (
            "v2 FLY125 pooled fit (8 cruise + 1 standby + 1 ramp window; drives all three "
            "FLY124 regimes through the trajectory sampler)"
        )
    else:
        label = "v2 FLY125 cruise fit (drives FLY124 standby/ramp through the trajectory sampler)"
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


def _named_fit(path: Path, *, rig: str, admissible: tuple[str, ...], role: str) -> dict[str, Any]:
    fit = json.loads(Path(path).read_text())
    if str(fit.get("schema")) not in FIT_SCHEMAS:
        die(f"{path}: schema {fit.get('schema')!r} is none of {list(FIT_SCHEMAS)}")
    named = " or ".join(repr(name) for name in admissible)
    if str(fit.get("support")) not in admissible:
        die(
            f"{path}: carries support {fit.get('support')!r}, but the {rig} arm's {role} fit is "
            f"defined on {named}"
        )
    fit["_path"] = str(path)
    return fit


def v2_regime_arm(
    rig: str,
    *,
    render_mod: Any,
    cruise_path: Path,
    standby_path: Path,
) -> Arm:
    """The PER-REGIME v2 arm: a standby-only fit and a cruise-only fit, gated.

    Selected by naming both files — ``--fit michaels=<cruise json> --fit
    michaels_standby=<standby json>``; there is no directory-selection route,
    because a per-regime candidate is a pair and a directory carries no
    pairing. Both are rendered on the support's own carrier by
    ``render.render_noise_regimes``, which uses the standby fit below
    ``rps_gating.STANDBY_MAX_RPS`` = 45 rev/s (slowest rotor), the cruise fit
    at or above ``CRUISE_MIN_RPS`` = 65 and the weight-interpolated power of
    the two in between — the previous generation's own regime thresholds, with
    its hard per-label export lookup replaced by the carrier itself.

    Only Michael's has a standby fit: DREGON is scored on cruise windows only.
    """
    if rig != "michaels":
        die(f"--fit {rig}_standby: only the michaels arm is defined per regime")
    for name in ("render_noise_regimes", "expected_periodogram_regimes"):
        if getattr(render_mod, name, None) is None:
            die(
                f"experiments.noise_model.render carries no {name}; this code is older than the "
                "per-regime composition and cannot score a per-regime candidate"
            )
    cruise = _named_fit(
        cruise_path, rig=rig, admissible=MICHAELS_FIT_SUPPORTS, role="cruise/pooled"
    )
    standby = _named_fit(
        standby_path, rig=rig, admissible=MICHAELS_STANDBY_FIT_SUPPORTS, role="standby"
    )
    return Arm(
        rig=rig,
        kind="v2",
        label=(
            "v2 FLY125 PER-REGIME pair: the standby-only fit below 45 rev/s, the cruise-only fit "
            "above 65, their powers interpolated across the ramp band (render_noise_regimes)"
        ),
        source=dict(
            composition="render.render_noise_regimes",
            per_regime={
                "standby": dict(support=standby.get("support"), path=standby["_path"]),
                "cruise": dict(support=cruise.get("support"), path=cruise["_path"]),
            },
            rule=(
                "per-sample sqrt-weight sum of the two regimes' renders, weight = "
                "rps_gating.regime_weight (smoothstep in the slowest rotor, 0 at 45 rev/s, 1 at "
                "65, settle_s disabled), so the composed power interpolates the two fitted "
                "regimes' levels across the ramp band"
            ),
        ),
        fit=cruise,
        regime_fits={"standby": standby, "cruise": cruise},
        _render_regimes=render_mod.render_noise_regimes,
        _spectrum_regimes=render_mod.expected_periodogram_regimes,
    )


def _comb_mean_check(fit: dict[str, Any], fits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Is the rendered DREGON comb really the declared mean of the four bench fits?

    The floor-only fit records the rule it froze its comb with in
    ``frozen_from.rule``; R1's rule is the LOG mean of the rates and scales, so
    both means are recomputed here and the one the fit declares is the
    deviation that counts. Nothing is assumed: a fit that names no rule is
    checked against the arithmetic mean and says so.
    """
    missing = [s for s in DREGON_COMB_SUPPORTS if s not in fits]
    if missing:
        return dict(available=False, missing_bench_fits=missing)
    declared = str(((fit.get("frozen_from") or {}).get("rule")) or "")
    kind = "log" if "log-mean" in declared else "arithmetic"
    params = [fits[s]["params"] for s in DREGON_COMB_SUPPORTS]
    rows: dict[str, Any] = {}
    for key in COMB_PARAM_KEYS:
        if key not in fit["params"] or any(key not in p for p in params):
            continue
        values = [float(p[key]) for p in params]
        mean = float(np.mean(values))
        log_mean = float(np.exp(np.mean(np.log(values)))) if all(v > 0.0 for v in values) else None
        reference = log_mean if kind == "log" and log_mean is not None else mean
        got = float(fit["params"][key])
        rows[key] = dict(
            bench_values=values,
            bench_mean=mean,
            bench_log_mean=log_mean,
            reference=reference,
            rendered=got,
            relative=(abs(got - reference) / abs(reference) if reference else None),
        )
    rel = [r["relative"] for r in rows.values() if r["relative"] is not None]
    return dict(
        available=bool(rows),
        bench_fits=list(DREGON_COMB_SUPPORTS),
        declared_rule=(declared or None),
        reference_mean=kind,
        per_parameter=rows,
        max_relative=(max(rel) if rel else None),
        rule=(
            "the floor-only flight fit must carry the bench comb mean unchanged, under the rule "
            "it declares in frozen_from.rule"
        ),
    )


def _fit_row(fit: dict[str, Any]) -> dict[str, Any]:
    opt = dict(fit.get("optimiser") or {})
    return dict(
        support=fit.get("support"),
        path=fit.get("_path"),
        mode=fit.get("mode"),
        n_fit_windows=len(fit.get("supports") or []),
        converged=opt.get("converged"),
        which_converged=opt.get("which_converged"),
        grad_norm=opt.get("grad_norm"),
        tol_nats_per_cell=opt.get("tol_nats_per_cell"),
        lbfgs_restart_gain_per_cell=opt.get("lbfgs_restart_gain_per_cell"),
        wall_s=opt.get("wall_s"),
        whittle_nats=(fit.get("objective") or {}).get("whittle_nats"),
    )


def fit_provenance(arm: Arm) -> dict[str, Any]:
    """What the candidate row must state about the fit it renders from.

    Convergence is NOT summarised away: the fit's own ``optimiser.converged``
    flag, which criterion fired, the final gradient norm, the tolerance it was
    tested against and the wall clock are carried into the round record so that
    every quoted gate number sits next to the status of the fit that produced
    it.

    A PER-REGIME arm renders from a PAIR, so it reports a row per regime and
    the summary a table can print without hiding either member: ``converged``
    is true only if BOTH fits converged, ``grad_norm`` and
    ``lbfgs_restart_gain_per_cell`` are the worst of the two, ``wall_s`` their
    sum, and ``whittle_nats`` is null because the two objectives are over
    different cells and are not addable.
    """
    if arm.kind != "v2" or arm.fit is None:
        return dict(kind=arm.kind, source=arm.source)
    if arm.regime_fits is not None:
        per = {regime: _fit_row(f) for regime, f in arm.regime_fits.items()}
        rows = list(per.values())

        def _worst(key: str) -> float | None:
            vals = [r[key] for r in rows if r.get(key) is not None]
            return max(float(v) for v in vals) if vals else None

        conv = [r["converged"] for r in rows]
        walls = [float(r["wall_s"]) for r in rows if r.get("wall_s") is not None]
        return dict(
            support="+".join(str(r["support"]) for r in rows),
            path=" + ".join(str(r["path"]) for r in rows),
            mode="+".join(sorted({str(r["mode"]) for r in rows})),
            n_fit_windows=sum(int(r["n_fit_windows"]) for r in rows),
            converged=(all(bool(c) for c in conv) if all(c is not None for c in conv) else None),
            which_converged={regime: r["which_converged"] for regime, r in per.items()},
            grad_norm=_worst("grad_norm"),
            tol_nats_per_cell=next(
                (r["tol_nats_per_cell"] for r in rows if r.get("tol_nats_per_cell") is not None),
                None,
            ),
            lbfgs_restart_gain_per_cell=_worst("lbfgs_restart_gain_per_cell"),
            wall_s=(sum(walls) if walls else None),
            whittle_nats=None,
            per_regime=per,
            composition=arm.source.get("composition"),
            composition_rule=arm.source.get("rule"),
        )
    return _fit_row(arm.fit) | dict(comb_mean_check=arm.source.get("comb_mean_check"))


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
    return row | dict(
        _clip=clip, _real=real, _render=renders[int(seeds[0])], _renders=renders, _arm=arm
    )


def _dump_audio(root: Path, support: GT.ScoredSupport, row: dict[str, Any]) -> None:
    """Write the scored audio of one support: the real clip and every seed.

    One ``.npz`` per support, float32, in the evaluator's absolute units and on
    the frozen timeline. This is the render the gates were read from, kept as
    evidence; it is not committed (the round record names the job that wrote
    it).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = support.key.replace("/", "_").replace(":", "_")
    arrays: dict[str, Any] = {"real": np.asarray(row["_real"], dtype=np.float32)}
    for seed, audio in row["_renders"].items():
        arrays[f"render_seed_{int(seed)}"] = np.asarray(audio, dtype=np.float32)
    arrays["rig"] = np.array(support.rig)
    arrays["regime"] = np.array(support.regime)
    arrays["sr"] = np.array(16000)
    np.savez(root / f"{name}.npz", **arrays)


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


# ── the two bars ────────────────────────────────────────────────────────────

#: The LEGACY PARITY bar of each rig: the previous best the v2 candidate has to
#: match. DREGON is the legacy synthetic cruise PIT MAE itself
#: (``gates.DREGON_BASELINE_PIT_MAE``); Michael's is 1.05 x the legacy
#: equal-regime mean, which is exactly the frozen Michael's gate bound
#: (``gates.MICHAELS_PIT_BOUND``).
PARITY_BAR: dict[str, float] = {
    "dregon": GT.DREGON_BASELINE_PIT_MAE,
    "michaels": GT.MICHAELS_PIT_BOUND,
}

#: The STRETCH bar. Only DREGON has one: the frozen 0.70-gap-closure target
#: ``real + 0.7 x (baseline - real)``, which is what ``gates.hppnet_gate``
#: itself applies. No stretch bar was ever defined for Michael's — its frozen
#: gate IS the parity bar — so it is reported as null, never invented.
STRETCH_BAR: dict[str, float | None] = {
    "dregon": GT.DREGON_PIT_TARGET,
    "michaels": None,
}


def _bar_row(value: float | None, bar: float | None) -> dict[str, Any]:
    within = None if (value is None or bar is None or not np.isfinite(value)) else value <= bar
    return dict(
        bar_rev_s=bar,
        within=within,
        margin_rev_s=(
            None if (value is None or bar is None or not np.isfinite(value)) else bar - value
        ),
    )


def bars(hppnet: dict[str, Any]) -> dict[str, Any]:
    """Both bars on the two HPPNet numbers, read off one frozen gate result.

    The frozen gate of ``gates.hppnet_gate`` applies the STRETCH bar on DREGON
    and the PARITY bar on Michael's; this function does not re-measure
    anything, it only reports each rig's number against both bars so that a
    candidate which reaches parity but not the stretch target is visible as
    such. A rig whose cohort is incomplete is ``within: null`` — never a pass.
    """
    if "dregon_cruise" not in hppnet:
        unavailable = str(hppnet.get("unavailable") or "the HPPNet probe did not run")
        return dict(
            unavailable=unavailable,
            dregon=None,
            michaels=None,
            parity_pass=False,
            stretch_pass=False,
        )
    d, m = hppnet["dregon_cruise"], hppnet["michaels_fly124"]
    d_mean = float(d["mean_candidate_rev_s"])
    d_upper = d["candidate_interval"].get("upper")
    d_complete = bool(d["checks"]["cohort_complete"] and d["checks"]["enough_clusters"])
    m_mean = float(m["rig_candidate_mae"])
    m_complete = bool(m["checks"]["cohort_complete"] and m["checks"]["all_regimes_present"])
    d_parity = _bar_row(d_mean if d_complete else None, PARITY_BAR["dregon"])
    d_stretch = _bar_row(d_mean if d_complete else None, STRETCH_BAR["dregon"])
    m_parity = _bar_row(m_mean if m_complete else None, PARITY_BAR["michaels"])
    dregon = dict(
        quantity="five-recording cruise PIT MAE mean, one-sided 95 % upper bound beside it",
        cohort_complete=d_complete,
        mean_rev_s=(d_mean if d_complete else None),
        interval_upper_rev_s=(d_upper if d_complete else None),
        parity=d_parity,
        parity_interval_upper=_bar_row(d_upper if d_complete else None, PARITY_BAR["dregon"]),
        stretch=d_stretch,
        stretch_interval_upper=_bar_row(d_upper if d_complete else None, STRETCH_BAR["dregon"]),
        frozen_gate_pass=bool(d["pass"]),
        missing_supports=list(d["missing_supports"]),
    )
    michaels = dict(
        quantity="equal-regime mean PIT MAE over standby/ramp/cruise (ratio of means)",
        cohort_complete=m_complete,
        mean_rev_s=(m_mean if m_complete else None),
        ratio=(float(m["aggregate_ratio"]) if m_complete else None),
        parity=m_parity,
        stretch=dict(
            bar_rev_s=None,
            within=None,
            margin_rev_s=None,
            note=(
                "no stretch bar is defined for Michael's: the frozen gate "
                f"(<= {GT.MICHAELS_RATIO_MAX} x {GT.MICHAELS_BASELINE_REGIME_MEAN:.6f}) IS the "
                "legacy parity bar"
            ),
        ),
        frozen_gate_pass=bool(m["pass"]),
        missing_supports=list(m["missing_supports"]),
    )
    parity_pass = bool(d_parity["within"] and m_parity["within"])
    stretch_pass = bool(d_stretch["within"] and m_parity["within"])
    return dict(
        rule=(
            "PARITY = the legacy previous-best bar on BOTH HPPNet numbers; STRETCH = the frozen "
            "0.70-gap DREGON target with Michael's parity bar unchanged"
        ),
        dregon=dregon,
        michaels=michaels,
        parity_pass=parity_pass,
        stretch_pass=stretch_pass,
    )


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
    candidate: str = "v2",
    dump_audio: Path | None = None,
    fit_files: dict[str, Path] | None = None,
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
            standby_path = (fit_files or {}).get(f"{rig}_standby")
            if standby_path is not None:
                cruise_path = (fit_files or {}).get(rig)
                if cruise_path is None:
                    die(
                        f"--fit {rig}_standby names the standby half of a per-regime candidate; "
                        f"--fit {rig}=<cruise fit json> must name the other half"
                    )
                assert cruise_path is not None
                arms[rig] = v2_regime_arm(
                    rig,
                    render_mod=render_mod,
                    cruise_path=cruise_path,
                    standby_path=standby_path,
                )
                continue
            assert fits_dir is not None
            arms[rig] = v2_arm(
                rig,
                fits_dir,
                render_mod=render_mod,
                spectrum_fn=spectrum_fn,
                fit_path=(fit_files or {}).get(rig),
            )
    probe = Probe.load() if with_probe else None
    payload: dict[str, Any] = dict(
        round=int(round_no),
        git=git_rev(),
        arm=dict(
            kind="legacy" if legacy else "v2",
            label={r: a.label for r, a in arms.items()},
            source={r: a.source for r, a in arms.items()},
        ),
        candidate=dict(
            name=str(candidate),
            rigs=list(rigs),
            fits={r: fit_provenance(a) for r, a in arms.items()},
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
        if dump_audio is not None:
            _dump_audio(dump_audio, support, row)
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
    payload["bars"] = bars(gates["hppnet"])
    payload["pass"] = bool(all(bool(g.get("pass")) for g in gates.values()))
    payload["frozen_gates_pass"] = payload["pass"]
    payload["parity_pass"] = bool(payload["bars"]["parity_pass"])
    payload["stretch_pass"] = bool(payload["bars"]["stretch_pass"])
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
    verdict = "PASS" if payload["pass"] else "FAIL"
    out: list[str] = [f"# Noise model v2 — round {payload['round']} gate score ({verdict})", ""]
    out.append(
        f"Arm: {payload['arm']['kind']} — "
        + "; ".join(f"{r}: {lbl}" for r, lbl in payload["arm"]["label"].items())
    )
    out.append(
        f"Render seeds {payload['protocol']['render_seeds']}, "
        f"{payload['protocol']['n_mics']} mics, git `{payload['git']}`."
    )
    out.append("")
    out += findings_body(payload)
    return "\n".join(out)


def findings_body(payload: dict[str, Any]) -> list[str]:
    """Every gate section of one scored record, from the verdict table down."""
    out: list[str] = []
    g = payload["gates"]
    out.append("## Gate verdicts")
    out.append("")
    out.append("| gate | verdict | number | threshold |")
    out.append("|---|---|---|---|")
    hp = g["hppnet"]
    if "dregon_cruise" in hp:
        d, m = hp["dregon_cruise"], hp["michaels_fly124"]
        measured_d = bool(d["clusters"])
        measured_m = any(r["n_blocks"] for r in m["per_regime"].values())
        out.append(
            f"| HPPNet DREGON cruise | {'PASS' if d['pass'] else ('FAIL' if measured_d else 'NOT RUN')} | "
            + (
                f"{d['mean_candidate_rev_s']:.6f} rev/s (95 % upper "
                f"{_fmt(d['candidate_interval']['upper'])})"
                if measured_d
                else f"no DREGON support measured ({len(d['missing_supports'])} missing)"
            )
            + f" | <= {d['target_rev_s']:.6f} |"
        )
        out.append(
            f"| HPPNet Michael's ratio | {'PASS' if m['pass'] else ('FAIL' if measured_m else 'NOT RUN')} | "
            + (
                f"{m['aggregate_ratio']:.6f} ({m['rig_candidate_mae']:.6f} rev/s)"
                if measured_m
                else f"no FLY124 support measured ({len(m['missing_supports'])} missing)"
            )
            + f" | <= {m['ratio_max']:.2f} ({m['point_bound_rev_s']:.6f} rev/s) |"
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
    if "dregon_cruise" in hp and not hp["dregon_cruise"]["clusters"]:
        d = hp["dregon_cruise"]
        out.append(
            "DREGON cruise: NOT RUN — no DREGON support was measured in this round "
            f"({len(d['missing_supports'])} of {len(GT.DREGON_CRUISE_SUPPORTS)} missing), so "
            f"neither the cluster mean nor its one-sided 95 % bound exists to put against the "
            f"frozen target {d['target_rev_s']:.6f} rev/s."
        )
        out.append("")
    elif "dregon_cruise" in hp:
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
            f"(relative {_fmt(d['real_arm']['relative'], '.3e')})."
        )
        out.append("")
    m = hp.get("michaels_fly124") if "dregon_cruise" in hp else None
    if m and any(r["n_blocks"] for r in m["per_regime"].values()):
        out.append("Michael's FLY124 per regime (rev/s):")
        out.append("")
        out.append("| regime | blocks | candidate | frozen baseline | ratio |")
        out.append("|---|---:|---:|---:|---:|")
        for regime in ("standby", "ramp", "cruise"):
            r = m["per_regime"][regime]
            out.append(
                f"| {regime} | {r['n_blocks']} | {_fmt(r['candidate_mae'])} | "
                f"{r['baseline_mae']:.6f} | {_fmt(r['ratio'], '.4f')} |"
            )
        out.append("")
        out.append(
            f"Equal-regime mean {m['rig_candidate_mae']:.6f} rev/s against the frozen baseline "
            f"{m['rig_baseline_mae']:.6f}; ratio {m['aggregate_ratio']:.6f} against the "
            f"{m['ratio_max']} bound ({m['arithmetic']})."
        )
    elif m:
        out.append(
            "Michael's FLY124: NOT RUN — no FLY124 support was measured in this round "
            f"({len(m['missing_supports'])} of {len(GT.MICHAELS_SUPPORTS)} missing)."
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
    return out


# ── composing the round record ──────────────────────────────────────────────

ROUND_SCHEMA = "noise-v2-round-score/1"
PROXY_GROUP: dict[str, str] = {"dregon": "dregon_cruise", "michaels": "michaels_cruise"}


def load_arm(path: Path) -> dict[str, Any]:
    """One scored arm record written by a ``run`` pass of this script."""
    d = json.loads(Path(path).read_text())
    for key in ("candidate", "gates", "measurements", "protocol"):
        if key not in d:
            die(f"{path}: not a scored arm record of this script (no {key!r})")
    d["_path"] = str(path)
    return d


def compose(
    *,
    round_no: int,
    arm_paths: list[Path],
    previous: Path | None,
    supports_index: Path | None,
    blocker: str | None,
    arm_jobs: dict[str, str] | None = None,
    arm_notes: dict[str, str] | None = None,
    compare: Path | None = None,
    not_run: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The round record: one candidate row per scored arm, gates recombined.

    Each arm was scored on its own rig (one remote job per rig), so the joint
    HPPNet gate is REEVALUATED here on the union of the arms' per-support PIT
    MAEs through :func:`gates.hppnet_gate` — the same frozen function, no
    re-measurement and no arithmetic of its own. The proxy gate's two groups
    are independent by construction, so each is carried from the arm that
    measured it; the likelihood gate exists only on Michael's cruise cells.
    Where several arms cover one rig, the FIRST listed is the primary one that
    enters the joint gate; the others stay as fully scored candidate rows.
    """
    arms = [load_arm(p) for p in arm_paths]
    for arm in arms:
        job = (arm_jobs or {}).get(str(arm["candidate"]["name"]))
        if job:
            _attribute_job(arm, job)
    primary: dict[str, dict[str, Any]] = {}
    for arm in arms:
        for rig in arm["candidate"]["rigs"]:
            primary.setdefault(rig, arm)
    pred: dict[str, float] = {}
    real: dict[str, float] = {}
    for rig, arm in primary.items():
        for key, row in arm["measurements"].items():
            if str(row["support"]["rig"]) != rig:
                continue
            if "pit_mae" in row:
                pred[key] = float(row["pit_mae"])
            if "real_pit_mae" in row:
                real[key] = float(row["real_pit_mae"])
    gates: dict[str, Any] = {}
    gates["hppnet"] = (
        GT.hppnet_gate(pred, real)
        if pred
        else dict(
            name="hppnet_pit_mae",
            status="not_run",
            unavailable="no arm carries a probed PIT MAE",
            **{"pass": False},
        )
    )
    groups: dict[str, Any] = {}
    proxy_template: dict[str, Any] = {}
    for rig, arm in primary.items():
        proxy = arm["gates"].get("proxy") or {}
        proxy_template = {k: v for k, v in proxy.items() if k not in ("groups", "pass")}
        blk = (proxy.get("groups") or {}).get(PROXY_GROUP[rig])
        if blk is not None:
            groups[PROXY_GROUP[rig]] = blk
    missing_groups = [g for g in PROXY_GROUP.values() if g not in groups]
    gates["proxy"] = dict(
        proxy_template,
        groups=groups,
        missing_groups=missing_groups,
        **{"pass": bool(groups and not missing_groups and all(b["pass"] for b in groups.values()))},
    )
    likelihood = next(
        (
            a["gates"]["likelihood"]
            for a in arms
            if "per_band" in (a["gates"].get("likelihood") or {})
        ),
        None,
    )
    if likelihood is None:
        likelihood = next(
            (a["gates"]["likelihood"] for a in arms if "michaels" in a["candidate"]["rigs"]),
            dict(
                name="likelihood_composite_risk",
                status="not_run",
                unavailable="no Michael's arm was scored, so no likelihood cell exists",
                **{"pass": False},
            ),
        )
    gates["likelihood"] = likelihood
    merged_bars = bars(gates["hppnet"])
    rows: list[dict[str, Any]] = []
    for arm in arms:
        cand = arm["candidate"]
        rows.append(
            dict(
                name=cand["name"],
                note=(arm_notes or {}).get(str(cand["name"])),
                rigs=cand["rigs"],
                primary_for=[r for r, a in primary.items() if a is arm],
                fits=cand["fits"],
                git=arm.get("git"),
                job=arm.get("job"),
                audio=arm.get("audio"),
                arm_record=arm["_path"],
                render_seeds=arm["protocol"]["render_seeds"],
                n_mics=arm["protocol"]["n_mics"],
                scorer=arm["protocol"].get("scorer"),
                bars=arm.get("bars"),
                hppnet=_candidate_hppnet(arm),
                proxy={
                    g: dict(
                        mean_ltas_abs_db=b["mean_ltas_abs_db"],
                        spread_ltas_abs_db=b["spread_ltas_abs_db"],
                        gate_db=b["gate_db"],
                        margin_db=b["margin_db"],
                        mean_mr_ltas=b["mean_mr_ltas"],
                        n_supports=b["n_supports"],
                        **{"pass": b["pass"]},
                    )
                    for g, b in ((arm["gates"].get("proxy") or {}).get("groups") or {}).items()
                    if b["n_supports"]
                },
                likelihood=_candidate_likelihood(arm),
                render_wall_s=sum(
                    float(e["render_seconds"])
                    for row in arm["measurements"].values()
                    for e in row["per_seed"]
                ),
            )
        )
    payload: dict[str, Any] = dict(
        schema=ROUND_SCHEMA,
        round=int(round_no),
        git=git_rev(),
        status="scored" if pred else "partially_scored",
        arm=dict(
            kind="v2",
            label={r: a["arm"]["label"].get(r, a["candidate"]["name"]) for r, a in primary.items()},
            source={r: a["arm"]["source"].get(r) for r, a in primary.items()},
        ),
        fits=sorted(
            {f["path"] for a in arms for f in a["candidate"]["fits"].values() if f.get("path")}
        ),
        candidates=rows,
        protocol=_composed_protocol(arms),
        measurements={k: v for a in arms for k, v in a["measurements"].items()},
        likelihood_cells=[c for a in arms for c in (a.get("likelihood_cells") or [])],
        gates=gates,
        bars=merged_bars,
    )
    payload["pass"] = bool(merged_bars["parity_pass"])
    payload["parity_pass"] = bool(merged_bars["parity_pass"])
    payload["stretch_pass"] = bool(merged_bars["stretch_pass"])
    payload["frozen_gates_pass"] = bool(all(bool(g.get("pass")) for g in gates.values()))
    payload["pass_rule"] = (
        "top-level pass = LEGACY PARITY on both HPPNet gates; stretch_pass is the frozen "
        "0.70-gap DREGON target; frozen_gates_pass additionally requires the proxy and "
        "likelihood gates"
    )
    if supports_index is not None and Path(supports_index).is_file():
        payload["support_availability"] = _support_availability(Path(supports_index))
    if previous is not None and Path(previous).is_file():
        prev = json.loads(Path(previous).read_text())
        if prev.get("legacy_smoke"):
            payload["legacy_smoke"] = prev["legacy_smoke"]
    if compare is not None and Path(compare).is_file():
        payload["previous_round"] = _round_over_round(Path(compare), payload)
    for rig, reason in (not_run or {}).items():
        if rig not in PROXY_GROUP:
            die(f"--not-run names {rig!r}, which is not a rig of this study")
        if rig in primary:
            die(f"--not-run names {rig!r}, but an arm of this round scored it")
        payload.setdefault("not_run", {})[rig] = str(reason)
    payload["blocker"] = blocker or _auto_blocker(payload, primary)
    return payload


ARTIFACT_BUCKET = "s3://omnirun-artifacts"


def _attribute_job(arm: dict[str, Any], job: str) -> None:
    """Name the omnirun job that scored this arm, and where its audio lives.

    The job id is only known to the submitter, never inside the job, so it is
    stamped here; the audio URI follows from the bucket layout
    ``<bucket>/<job>/outputs/<repo-relative path>`` the job's outputs are
    collected into.
    """
    arm["job"] = str(job)
    audio = dict(arm.get("audio") or {})
    if audio.get("dir"):
        audio["uri"] = f"{ARTIFACT_BUCKET}/{job}/outputs/{str(audio['dir']).lstrip('./')}"
        arm["audio"] = audio


def _candidate_hppnet(arm: dict[str, Any]) -> dict[str, Any]:
    """The candidate row's own HPPNet numbers, on the rigs it was scored on."""
    hp = arm["gates"].get("hppnet") or {}
    if "dregon_cruise" not in hp:
        return dict(status="not_run", reason=hp.get("unavailable"))
    out: dict[str, Any] = {}
    for rig, key in (("dregon", "dregon_cruise"), ("michaels", "michaels_fly124")):
        if rig not in arm["candidate"]["rigs"]:
            continue
        blk = hp[key]
        out[rig] = (
            dict(
                mean_rev_s=blk["mean_candidate_rev_s"],
                interval=blk["candidate_interval"],
                real_arm_mean_rev_s=blk["real_arm"]["mean_rev_s"],
                real_arm_relative=blk["real_arm"]["relative"],
                frozen_gate_pass=blk["pass"],
                per_support={c["support"]: c["candidate"] for c in blk["clusters"]},
            )
            if rig == "dregon"
            else dict(
                rig_mean_rev_s=blk["rig_candidate_mae"],
                ratio=blk["aggregate_ratio"],
                per_regime={
                    r: dict(
                        candidate_mae=v["candidate_mae"],
                        baseline_mae=v["baseline_mae"],
                        ratio=v["ratio"],
                        n_blocks=v["n_blocks"],
                    )
                    for r, v in blk["per_regime"].items()
                },
                frozen_gate_pass=blk["pass"],
            )
        )
    return out


def _candidate_likelihood(arm: dict[str, Any]) -> dict[str, Any]:
    lk = arm["gates"].get("likelihood") or {}
    if "per_band" not in lk:
        return dict(status="not_run", reason=lk.get("unavailable"))
    return dict(
        decisive_band=lk["decisive_band"],
        per_band={
            b: dict(
                model_nats_per_s=lk["per_band"][b]["model_nats_per_s"],
                oracle_nats_per_s=lk["per_band"][b]["oracle_nats_per_s"],
                margin_nats_per_s=lk["per_band"][b]["margin_nats_per_s"],
                below_oracle=lk["per_band"][b]["below_oracle"],
            )
            for b in ("comb", "floor", "full")
        },
        **{"pass": lk["pass"]},
    )


def _composed_protocol(arms: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = sorted({int(s) for a in arms for s in a["protocol"]["render_seeds"]})
    mics = sorted({int(a["protocol"]["n_mics"]) for a in arms})
    scorer = next((a["protocol"]["scorer"] for a in arms if a["protocol"].get("scorer")), None)
    return dict(
        render_seeds=seeds,
        n_mics=(mics[0] if len(mics) == 1 else mics),
        hppnet=dict(
            experiment=GT.SCORER_EXPERIMENT,
            checkpoint=GT.SCORER_CKPT,
            sha256=GT.SCORER_SHA256,
            verified=bool(scorer and scorer.get("sha256") == GT.SCORER_SHA256),
            record=scorer,
        ),
        proxy="ltas_abs_db on mic 0; mr_ltas report-only",
        likelihood=dict(
            n_fft=GT.LIKELIHOOD_N_FFT,
            hop=GT.LIKELIHOOD_HOP,
            pool="FLY124 cruise supports",
            band_edge_hz=GT.BAND_EDGE_HZ,
            decisive_band="comb",
        ),
        note=next(a["protocol"].get("note") for a in arms),
    )


def _support_availability(index: Path) -> dict[str, Any]:
    d = json.loads(index.read_text())
    sets = {
        name: dict(required=int(blk["n_specs"]), materialized=int(blk["n_built"]))
        for name, blk in (d.get("sets") or {}).items()
    }
    return dict(
        source=str(index),
        n_supports_materialized=int(d.get("n_supports") or 0),
        sets=sets,
        updated=d.get("updated"),
    )


#: The gate scalars compared round over round: where each lives in a round
#: record, and which direction counts as an improvement. ``lower`` is the
#: normal case (an error); the likelihood comb margin is model-minus-oracle, so
#: a LESS negative number is the better one.
ROUND_QUANTITIES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "dregon cruise PIT MAE (5-recording mean)",
        "rev/s",
        "lower",
        ("bars", "dregon", "mean_rev_s"),
    ),
    (
        "dregon cruise PIT MAE (95 % upper)",
        "rev/s",
        "lower",
        ("bars", "dregon", "interval_upper_rev_s"),
    ),
    ("michaels equal-regime PIT MAE", "rev/s", "lower", ("bars", "michaels", "mean_rev_s")),
    ("michaels ratio vs the legacy regime mean", "x", "lower", ("bars", "michaels", "ratio")),
    (
        "proxy ltas_abs_db, dregon cruise",
        "dB",
        "lower",
        ("gates", "proxy", "groups", "dregon_cruise", "mean_ltas_abs_db"),
    ),
    (
        "proxy ltas_abs_db, michaels cruise",
        "dB",
        "lower",
        ("gates", "proxy", "groups", "michaels_cruise", "mean_ltas_abs_db"),
    ),
    (
        "mr_ltas, dregon cruise (report-only)",
        "—",
        "lower",
        ("gates", "proxy", "groups", "dregon_cruise", "mean_mr_ltas"),
    ),
    (
        "mr_ltas, michaels cruise (report-only)",
        "—",
        "lower",
        ("gates", "proxy", "groups", "michaels_cruise", "mean_mr_ltas"),
    ),
    (
        "likelihood comb-band margin vs oracle",
        "nats/s",
        "higher",
        ("gates", "likelihood", "per_band", "comb", "margin_nats_per_s"),
    ),
)


def _dig(record: dict[str, Any], path: tuple[str, ...]) -> float | None:
    """One scalar of a round record, or ``None`` where the round did not run it."""
    node: Any = record
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    if node is None or isinstance(node, bool):
        return None
    try:
        value = float(node)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _round_over_round(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """This round's gate scalars beside the previous round's, with the delta.

    Nothing is recomputed: both sides are read out of the two round records as
    written, and a quantity absent from either round stays ``None`` rather than
    being filled in from anywhere else.
    """
    prev = json.loads(Path(path).read_text())
    rows: list[dict[str, Any]] = []
    for label, unit, direction, keys in ROUND_QUANTITIES:
        was, now = _dig(prev, keys), _dig(payload, keys)
        delta = (now - was) if (was is not None and now is not None) else None
        rows.append(
            dict(
                quantity=label,
                unit=unit,
                better=direction,
                previous=was,
                current=now,
                delta=delta,
                improved=(
                    None
                    if delta is None or delta == 0.0
                    else bool((delta < 0.0) if direction == "lower" else (delta > 0.0))
                ),
            )
        )
    return dict(
        source=str(path),
        round=prev.get("round"),
        git=prev.get("git"),
        candidates=[c.get("name") for c in (prev.get("candidates") or [])],
        quantities=rows,
    )


def _auto_blocker(payload: dict[str, Any], primary: dict[str, dict[str, Any]]) -> str:
    parts: list[str] = []
    for rig in ("dregon", "michaels"):
        if rig not in primary:
            parts.append(f"no {rig} arm was scored")
    for name, gate in payload["gates"].items():
        if gate.get("status") == "not_run":
            parts.append(f"the {name} gate is not_run: {gate.get('unavailable')}")
    missing = payload["gates"]["proxy"].get("missing_groups") or []
    if missing:
        parts.append("proxy groups without any measured support: " + ", ".join(missing))
    if not parts:
        return (
            "none: every gate of this round was evaluated on rendered candidate audio and the "
            "frozen probe."
        )
    return "; ".join(parts) + "."


def compose_findings(payload: dict[str, Any], *, provenance: str | None) -> str:
    parity = "PASS" if payload["parity_pass"] else "FAIL"
    stretch = "PASS" if payload["stretch_pass"] else "FAIL"
    out: list[str] = [
        f"# Noise model v2 — round {payload['round']} gate score: "
        f"PARITY {parity} / STRETCH {stretch}",
        "",
        f"Top-level pass (legacy parity on both HPPNet gates): **{str(payload['pass']).lower()}**. "
        f"Stretch (frozen 0.70-gap DREGON target): **{str(payload['stretch_pass']).lower()}**. "
        f"All three frozen gates: **{str(payload['frozen_gates_pass']).lower()}**. "
        f"Record `results/noise_v2/rounds/round{payload['round']}.json`, "
        f"git `{str(payload['git'])[:12]}`.",
        "",
        "## Candidates",
        "",
        "| candidate | rig(s) | fit JSON | converged | fit wall (s) | render+probe job |",
        "|---|---|---|---|---:|---|",
    ]
    for row in payload["candidates"]:
        for rig, fit in row["fits"].items():
            conv = fit.get("converged")
            label = (
                "—"
                if conv is None
                else (
                    "yes"
                    if conv
                    else f"**NO** ({fit.get('which_converged')}, |grad| {_fmt(fit.get('grad_norm'), '.0f')})"
                )
            )
            out.append(
                f"| `{row['name']}` | {rig} | `{fit.get('path')}` | {label} | "
                f"{_fmt(fit.get('wall_s'), '.0f')} | `{row.get('job') or '—'}` |"
            )
    notes = [(row["name"], row["note"]) for row in payload["candidates"] if row.get("note")]
    if notes:
        out.append("")
        for name, note in notes:
            out.append(f"* `{name}` — {note}")
    out.append("")
    out.append("## The two bars")
    out.append("")
    out.append("| rig | gate quantity | number | PARITY bar | parity | STRETCH bar | stretch |")
    out.append("|---|---|---:|---:|---|---:|---|")
    b = payload["bars"]
    not_run = payload.get("not_run") or {}
    for rig in ("dregon", "michaels"):
        blk = b.get(rig)
        absent = f"NOT RUN ({not_run[rig]})" if rig in not_run else "not run"
        if not blk:
            out.append(
                f"| {rig} | — | {absent} | {_fmt(PARITY_BAR[rig])} | — | "
                f"{_fmt(STRETCH_BAR[rig])} | — |"
            )
            continue
        out.append(
            f"| {rig} | {blk['quantity']} | "
            f"{absent if blk['mean_rev_s'] is None else _fmt(blk['mean_rev_s'])} | "
            f"{_fmt(blk['parity']['bar_rev_s'])} | {_verdict(blk['parity']['within'])} "
            f"(margin {_fmt(blk['parity']['margin_rev_s'], '+.6f')}) | "
            f"{_fmt(blk['stretch']['bar_rev_s'])} | {_verdict(blk['stretch']['within'])} "
            f"(margin {_fmt(blk['stretch']['margin_rev_s'], '+.6f')}) |"
        )
    out.append("")
    out.append(
        "PARITY is the legacy previous best: DREGON synthetic cruise PIT MAE "
        f"{GT.DREGON_BASELINE_PIT_MAE:.6f} rev/s (real arm {GT.DREGON_REAL_PIT_MAE:.6f}), "
        f"Michael's {GT.MICHAELS_RATIO_MAX} x {GT.MICHAELS_BASELINE_REGIME_MEAN:.6f} = "
        f"{GT.MICHAELS_PIT_BOUND:.6f} rev/s. STRETCH exists only on DREGON: "
        f"{GT.DREGON_PIT_TARGET:.6f} = real + {GT.DREGON_GAP_FRACTION} x (baseline - real); "
        "Michael's frozen gate IS its parity bar."
    )
    for rig, reason in not_run.items():
        out.append("")
        out.append(f"`{rig}`: NOT RUN ({reason}). Every {rig} cell of this record is a dash.")
    out.append("")
    rr = payload.get("previous_round")
    if rr:
        out.append(f"## Round {rr.get('round')} → round {payload['round']}")
        out.append("")
        out.append(
            f"| quantity | unit | round {rr.get('round')} | round {payload['round']} | "
            f"Δ (R{payload['round']} − R{rr.get('round')}) | direction | moved |"
        )
        out.append("|---|---|---:|---:|---:|---|---|")
        for q in rr["quantities"]:
            spec = ".4f" if q["unit"] in ("dB", "x", "—") else ".6f"
            moved = (
                "—" if q["improved"] is None else ("**better**" if q["improved"] else "**worse**")
            )
            out.append(
                f"| {q['quantity']} | {q['unit']} | {_fmt(q['previous'], spec)} | "
                f"{_fmt(q['current'], spec)} | "
                f"{'—' if q['delta'] is None else format(float(q['delta']), '+' + spec)} | "
                f"{q['better']} is better | {moved} |"
            )
        out.append("")
        out.append(
            f"Both sides are read verbatim out of `{rr.get('source')}` (git "
            f"`{str(rr.get('git'))[:12]}`, candidates "
            + ", ".join(f"`{c}`" for c in (rr.get("candidates") or []))
            + ") and this record; a dash is a quantity one of the two rounds did not measure, "
            "never a substituted number."
        )
        out.append("")
    out.append("## Per-candidate numbers")
    out.append("")
    out.append(
        "| candidate | HPPNet | proxy `ltas_abs_db` (gate) | `mr_ltas` | likelihood comb margin |"
    )
    out.append("|---|---:|---:|---:|---:|")
    for row in payload["candidates"]:
        hp = row["hppnet"]
        if hp.get("status") == "not_run":
            hp_txt = "not run"
        elif "dregon" in hp:
            hp_txt = (
                f"{_fmt(hp['dregon']['mean_rev_s'])} rev/s "
                f"(95 % upper {_fmt(hp['dregon']['interval'].get('upper'))})"
            )
        else:
            hp_txt = (
                f"{_fmt(hp['michaels']['rig_mean_rev_s'])} rev/s "
                f"(ratio {_fmt(hp['michaels']['ratio'], '.4f')})"
            )
        proxy = row["proxy"]
        proxy_txt = "; ".join(
            f"{_fmt(v['mean_ltas_abs_db'], '.4f')} ({_fmt(v['gate_db'], '.4f')})"
            for v in proxy.values()
        )
        mr_txt = "; ".join(_fmt(v["mean_mr_ltas"], ".4f") for v in proxy.values())
        lk = row["likelihood"]
        lk_txt = (
            "not run"
            if lk.get("status") == "not_run"
            else f"{lk['per_band']['comb']['margin_nats_per_s']:+,.4f} nats/s"
        )
        out.append(
            f"| `{row['name']}` | {hp_txt} | {proxy_txt or '—'} | {mr_txt or '—'} | {lk_txt} |"
        )
    out.append("")
    for row in payload["candidates"]:
        audio = row.get("audio") or {}
        if audio:
            out.append(
                f"Rendered audio of `{row['name']}` (uncommitted): `{audio.get('uri') or '—'}` "
                f"(job `{row.get('job') or '—'}`, written to `{audio.get('dir')}` in the job's "
                "worktree)."
            )
    out.append("")
    out += findings_body(payload)
    smoke = payload.get("legacy_smoke")
    if smoke:
        out.append("## Legacy smoke comparability (recorded earlier, NOT re-run)")
        out.append("")
        out.append(
            f"| legacy replay (`{smoke.get('job')}`) | observed | frozen scalar | relative delta |"
        )
        out.append("|---|---:|---:|---:|")
        for label, key in (
            ("DREGON cruise synthetic PIT MAE", "dregon_cruise_pit_mae"),
            ("Michael's equal-regime synthetic PIT MAE", "michaels_equal_regime_pit_mae"),
        ):
            blk = smoke.get(key) or {}
            out.append(
                f"| {label} | {_fmt(blk.get('observed'))} | {_fmt(blk.get('frozen'))} | "
                f"{_fmt(blk.get('relative_delta'), '.2e')} |"
            )
        out.append("")
        out.append(f"Status `{smoke.get('status')}`: {smoke.get('reason')}")
        out.append("")
    out.append("## Exact blocker")
    out.append("")
    out.append(str(payload.get("blocker")))
    out.append("")
    if provenance:
        out.append(provenance.strip())
        out.append("")
        out.append("### Addendum — this scoring pass")
        out.append("")
        out.append(
            "The `Provenance` section above is kept verbatim from the earlier, unscored record; "
            "the gates it calls `not_run` are the ones measured here, and whatever is still not "
            "run is named in `Exact blocker`. This pass:"
        )
        out.append("")
        for row in payload["candidates"]:
            out.append(
                f"* `{row['name']}` — fit git `{str(row.get('git'))[:12]}`, render/probe job "
                f"`{row.get('job') or 'local'}`, arm record `{row['arm_record']}`, render wall "
                f"{_fmt(row.get('render_wall_s'), '.0f')} s."
            )
        out.append("")
    else:
        out += _own_provenance(payload)
    return "\n".join(out)


def _own_provenance(payload: dict[str, Any]) -> list[str]:
    """The round's OWN provenance, when there is no earlier findings to carry.

    Round 1 inherited a `## Provenance` section from the unscored record it
    overwrote; a fresh round has none, so the section is written from the record
    itself: the frozen definitions' source, the probe checkpoint and whether its
    digest was verified in the job, and one line per candidate naming the fit it
    renders from, the fit's convergence status, the omnirun job that scored it
    and where its rendered audio lives.
    """
    hp = payload["protocol"]["hppnet"]
    out: list[str] = ["## Provenance", ""]
    out.append(
        "The frozen support identities, render seeds, one-sided interval, regime-mean ratio, "
        "proxy thresholds, probe checkpoint digest and likelihood/oracle definition are all "
        "read from `src/experiments/noise_model/gates.py`; this record adds no scalar of its "
        f"own. Probe `{hp['experiment']}/{hp['checkpoint']}`, sha256 `{hp['sha256']}`, "
        f"verified in-job: **{str(bool(hp.get('verified'))).lower()}** (the scoring pass dies "
        "on a digest mismatch, so a scored arm cannot exist without the match). Render seeds "
        f"{payload['protocol']['render_seeds']}, {payload['protocol']['n_mics']} microphones. "
        f"Record git `{str(payload['git'])[:12]}`."
    )
    out.append("")
    out.append("| candidate | rig(s) | fit | fit git | converged | job | audio |")
    out.append("|---|---|---|---|---|---|---|")
    for row in payload["candidates"]:
        for rig, fit in row["fits"].items():
            conv = fit.get("converged")
            out.append(
                f"| `{row['name']}` | {rig} | `{fit.get('path')}` | "
                f"`{str(row.get('git'))[:12]}` | "
                f"{'—' if conv is None else ('yes' if conv else '**NO**')} | "
                f"`{row.get('job') or 'local'}` | "
                f"`{(row.get('audio') or {}).get('uri') or '—'}` |"
            )
    out.append("")
    for row in payload["candidates"]:
        out.append(
            f"* `{row['name']}` — arm record `{row['arm_record']}`, render wall "
            f"{_fmt(row.get('render_wall_s'), '.0f')} s, "
            f"{len(row['rigs'])} rig(s) {', '.join(row['rigs'])}, primary for "
            f"{', '.join(row['primary_for']) or 'nothing'}."
        )
    avail = payload.get("support_availability")
    if avail:
        sets = ", ".join(
            f"{name} {blk['materialized']}/{blk['required']}"
            for name, blk in (avail.get("sets") or {}).items()
        )
        out.append("")
        out.append(
            f"Support availability from `{avail.get('source')}` "
            f"(updated {avail.get('updated')}): {sets}."
        )
    out.append("")
    return out


def _verdict(within: Any) -> str:
    return "—" if within is None else ("**PASS**" if within else "**FAIL**")


def carried_provenance(path: Path) -> str | None:
    """The ``## Provenance`` section of the previous findings, kept verbatim."""
    if not Path(path).is_file():
        return None
    text = Path(path).read_text()
    marker = "\n## Provenance"
    idx = text.find(marker)
    return None if idx < 0 else text[idx + 1 :]


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
    ap.add_argument(
        "--fits", type=Path, default=None, help=f"directory of fit JSONs {list(FIT_SCHEMAS)}"
    )
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
    ap.add_argument(
        "--candidate",
        default=None,
        help="name of this arm's candidate row in the round record (default: the --out-tag)",
    )
    ap.add_argument(
        "--arm-out",
        type=Path,
        default=None,
        help="write the scored arm record here instead of round<N>[_tag].json",
    )
    ap.add_argument(
        "--fit",
        action="append",
        default=[],
        metavar="RIG[_standby]=PATH",
        help=(
            "render that rig from this exact fit file instead of the one --fits selects by "
            "support name (a retry fit shares its support name with the fit it retries). "
            "'--fit michaels_standby=PATH' additionally makes the arm PER-REGIME: the standby "
            "fit is rendered below 45 rev/s, the '--fit michaels=' cruise fit above 65, and "
            "their powers are interpolated across the ramp band"
        ),
    )
    ap.add_argument(
        "--dump-audio",
        type=Path,
        default=None,
        help="write one .npz per support (real clip plus every render seed) into this directory",
    )
    ap.add_argument("--job", default=None, help="the omnirun job id this pass runs under")
    ap.add_argument(
        "--candidate-note",
        action="append",
        default=[],
        metavar="CANDIDATE=TEXT",
        help=(
            "with --compose: a caveat printed under that candidate's row (e.g. a defective "
            "frozen input that makes its numbers a pipeline proof, not a parity result)"
        ),
    )
    ap.add_argument(
        "--audio-uri", default=None, help="where the dumped audio is retrievable (s3 URI)"
    )
    ap.add_argument(
        "--compose",
        type=Path,
        nargs="+",
        default=None,
        metavar="ARM_JSON",
        help=(
            "compose the round record from scored arm records instead of scoring: the first "
            "record covering a rig is the primary one that enters the joint frozen gate"
        ),
    )
    ap.add_argument(
        "--supports-index",
        type=Path,
        default=None,
        help="with --compose: the supports index whose availability is recorded",
    )
    ap.add_argument(
        "--blocker",
        default=None,
        help="with --compose: the blocker sentence (default: derived from what was not run)",
    )
    ap.add_argument(
        "--arm-job",
        action="append",
        default=[],
        metavar="CANDIDATE=JOB",
        help=(
            "with --compose: the omnirun job that scored that candidate (the job id is not "
            "visible inside the job); the dumped audio URI follows from it"
        ),
    )
    ap.add_argument(
        "--previous",
        type=Path,
        default=None,
        help=(
            "with --compose: the round record whose legacy-smoke row is carried forward "
            "verbatim (default: this round's own earlier record, which a first pass has not "
            "written yet)"
        ),
    )
    ap.add_argument(
        "--compare",
        type=Path,
        default=None,
        help=(
            "with --compose: the previous round's record; its gate scalars enter the "
            "round-over-round delta table beside this round's"
        ),
    )
    ap.add_argument(
        "--not-run",
        action="append",
        default=[],
        metavar="RIG=REASON",
        help=(
            "with --compose: why that rig has no arm in this round (e.g. "
            "michaels='fit in flight: <job>'); printed where its number would be, so an "
            "absent rig is never a silent dash"
        ),
    )
    args = ap.parse_args(argv)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    out_json = Path(args.out) / f"round{int(args.round)}{tag}.json"
    out_dir = Path(args.out) / f"round{int(args.round)}{tag}" / "score"
    if args.compose:
        previous = args.previous or (out_json if out_json.is_file() else None)
        provenance = carried_provenance(out_dir / "findings.md")
        payload = compose(
            round_no=int(args.round),
            arm_paths=[Path(p) for p in args.compose],
            previous=previous,
            supports_index=args.supports_index,
            blocker=args.blocker,
            compare=args.compare,
            not_run=dict(
                str(spec).split("=", 1)
                for spec in args.not_run  # type: ignore[misc]
            ),
            arm_jobs=dict(
                str(spec).split("=", 1)
                for spec in args.arm_job  # type: ignore[misc]
            ),
            arm_notes=dict(
                str(spec).split("=", 1)
                for spec in args.candidate_note  # type: ignore[misc]
            ),
        )
        RE.write_json(out_json, payload)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "findings.md").write_text(compose_findings(payload, provenance=provenance))
        print(f"wrote {out_json} and {out_dir / 'findings.md'}")
        print(
            f"round {payload['round']}: parity "
            f"{'PASS' if payload['parity_pass'] else 'FAIL'}, stretch "
            f"{'PASS' if payload['stretch_pass'] else 'FAIL'}"
        )
        return 0
    legacy = args.legacy_export is not None
    if not legacy and args.fits is None:
        die("pass --fits <dir> with the round's fit JSONs, or --legacy-export baseline")
    rigs = tuple(str(args.rigs).replace(",", " ").split())
    fit_files = {str(spec).split("=", 1)[0]: Path(str(spec).split("=", 1)[1]) for spec in args.fit}
    for key, path in fit_files.items():
        if key not in rigs and key.removesuffix("_standby") not in rigs:
            die(
                f"--fit {key}=...: {key!r} is neither a scored rig {list(rigs)} nor '<rig>_standby'"
            )
        if not Path(path).is_file():
            die(f"--fit {key}={path}: no such file")
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
        rigs=rigs,
        with_probe=not args.no_probe,
        candidate=str(args.candidate or args.out_tag or ("legacy" if legacy else "v2")),
        dump_audio=args.dump_audio,
        fit_files=fit_files,
    )
    if args.job:
        payload["job"] = str(args.job)
    if args.dump_audio or args.audio_uri:
        payload["audio"] = dict(
            dir=(str(args.dump_audio) if args.dump_audio else None), uri=args.audio_uri
        )
    if args.arm_out:
        out_json = Path(args.arm_out)
        out_dir = out_json.parent
    RE.write_json(out_json, payload)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "findings.md").write_text(findings(payload))
    print(f"wrote {out_json} and {out_dir / 'findings.md'}")
    print(f"round {payload['round']}: {'PASS' if payload['pass'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
