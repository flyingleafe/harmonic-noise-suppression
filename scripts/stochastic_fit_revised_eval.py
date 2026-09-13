"""The revised-phase gate runner: baseline calibration, then candidate checks.

    python scripts/stochastic_fit_revised_eval.py --manifest <frozen-manifest> --plan
    python scripts/stochastic_fit_revised_eval.py --manifest <frozen-manifest> --prepare
    python scripts/stochastic_fit_revised_eval.py --manifest <frozen-manifest> --check

``--plan`` resolves the cohorts, the supports and the family coverage without
scoring anything. ``--prepare`` measures the BASELINE arm: it freezes the family
choice on calibration supports, calibrates the real-vs-synthetic gap on the
held-out supports, and calibrates the null-variation tolerances from the frozen
render seeds alone — at recording level (DREGON clusters) and at block level
(Michael's, conditional on FLY124). ``--check`` measures the candidate arm on
exactly the same window identities with the same frozen seeds, applies every
frozen gate, writes the metrics and the decisions, and exits nonzero when any
gate is unmet. ``--verify-adapter`` checks a predicted-``M`` adapter against its
own renderer's mean statistics.

Gate-integrity properties, deliberately built in:

* **Thresholds come from the manifest or the calibration, never from this
  file.** A missing threshold is a failed gate or a loud exit, never a
  permissive default, so a tolerance cannot be retro-fitted after a candidate
  has been seen.
* **The frozen record covers the referenced artifacts**, not just the manifest
  text: the manifest SHA-256, the scorer checkpoint SHA-256, every baseline and
  candidate export SHA-256 and every RESOLVED dataset version travel into
  ``calibration.json`` and are re-verified by ``--check``.
* **The frozen cohort is required in full.** A missing or insufficient support
  fails a gate before anything is bootstrapped; the cohort never shrinks to
  whichever recordings happened to work.
* **Every arm is measured against the same raw telemetry array**, on the same
  window identities, with the same frozen render seeds, on the regime's
  PRE-REGISTERED support, at the exact frozen sample length and microphone set.
* **Model families do not share a law.** A legacy export goes through the
  legacy renderer and the historical-forward adapter; a ``model_family``
  export goes through ``revised_phase.render_revised`` /
  ``revised_phase.predict_spectrum``. Each door refuses the other's file.
* **No candidate is measured before its provenance is checked**: the
  candidate's ``training_provenance.clips`` must not intersect any scored
  support, and its fit manifest digest is pinned.

Reused rather than reimplemented: ``scripts/_synthetic_probe.py::score`` (the
zoo/HPPNet/PIT path ``scripts/_stage2_probe.py`` already imports),
``experiments.stochastic_fit.clips`` for every real clip,
``stage2.render_from_export`` for legacy arms, ``revised_phase`` for the
candidate, ``phase_kernel``'s frozen front-end constants, and
``accept_stats.BANDS`` for the LTAS bands.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiments.stochastic_fit import clips as C
from experiments.stochastic_fit import revised_eval as RE
from experiments.stochastic_fit import revised_phase as RP
from experiments.stochastic_fit import stage2 as S2
from experiments.stochastic_fit.data import Clip

REQUIRED_GATE_KEYS = (
    "alpha",
    "bootstrap_seed",
    "dregon_gap_fraction",
    "michaels_ratio_max",
    "composite_tolerance",
)
#: The LTAS needs one 8192-point Welch window inside the scored support.
LTAS_N = 8192


def die(message: str) -> None:
    raise SystemExit(f"error: {message}")


def require(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d or d[key] is None:
        die(f"{ctx}: required field {key!r} is missing — the frozen manifest must state it")
    return d[key]


def probe_helpers() -> Any:
    """``scripts/_synthetic_probe.py``, imported by path and reused as-is."""
    path = Path(__file__).resolve().parent / "_synthetic_probe.py"
    spec = importlib.util.spec_from_file_location("_synthetic_probe", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def checkpoint_record(experiment: str, ckpt: str) -> dict[str, Any]:
    """Pin the scorer: the resolved checkpoint file and its SHA-256 digest."""
    from hydra import compose, initialize_config_dir

    from training.config import register_configs
    from utils.checkpoints import resolve_checkpoint_uri
    from zoo.cache import REPO_ROOT
    from zoo.frame_model import _checkpoint_ref

    register_configs()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])
    ref = _checkpoint_ref(experiment, ckpt, cfg)
    local = Path(resolve_checkpoint_uri(ref, REPO_ROOT / ".cache" / "r2_checkpoints"))
    return dict(
        experiment=experiment,
        ckpt=ckpt,
        reference=ref,
        local_path=str(local),
        bytes=local.stat().st_size,
        sha256=hashlib.sha256(local.read_bytes()).hexdigest(),
    )


# ── the manifest and the frozen input record ────────────────────────────────


def load_manifest(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    man = json.loads(raw.decode())
    if str(man.get("schema")) != RE.SCHEMA:
        die(f"{path}: schema {man.get('schema')!r}, expected {RE.SCHEMA!r}")
    gates = dict(require(man, "gates", str(path)))
    for key in REQUIRED_GATE_KEYS:
        require(gates, key, f"{path}: gates")
    require(man, "scorer", str(path))
    require(man, "cohorts", str(path))
    require(man, "arms", str(path))
    require(man, "null_variation", str(path))
    temp = man.get("composite_temperature")
    if temp is not None:
        for key in ("master_seed", "B"):
            if key not in temp or temp[key] is None:
                die(f"{path}: composite_temperature.{key} is required once the section is present")
        if int(temp["B"]) < 2:
            die(f"{path}: composite_temperature.B must be at least 2")
    man["_digest"] = hashlib.sha256(raw).hexdigest()
    man["_path"] = str(path)
    man["gates"] = gates
    return man


def _canonical_manifest(man: dict[str, Any]) -> dict[str, Any]:
    """Deep copy with run-input and authoring keys removed.

    The protocol fingerprint covers every design decision — gates,
    observations, cohorts, baseline families, seeds — but excludes only the
    run inputs that change between --prepare and --check:
    ``arms.candidate``, ``candidate_arm_template``, ``out_dir``,
    ``calibration_path``, and authoring/status/runtime keys.
    """
    import copy

    canon = copy.deepcopy(man)
    for key in ("_digest", "_path", "status", "notes", "out_dir", "calibration_path"):
        canon.pop(key, None)
    arms = canon.get("arms")
    if isinstance(arms, dict) and "candidate" in arms:
        arms.pop("candidate", None)
    canon.pop("candidate_arm_template", None)
    return canon


def protocol_fingerprint(man: dict[str, Any]) -> str:
    """SHA-256 of the canonical protocol content of a manifest."""
    canon = _canonical_manifest(man)
    raw = json.dumps(canon, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


_EXPORTS: dict[tuple[str, str, str], RE.ExportBundle] = {}


def read_export_cached(
    path: str, *, family: str, regime: str, declared: dict[str, Any] | None
) -> RE.ExportBundle:
    key = (str(path), family, regime)
    if key not in _EXPORTS:
        _EXPORTS[key] = RE.read_export(path, family=family, regime=regime, declared=declared)
    return _EXPORTS[key]


def cohort_families(cohort: dict[str, Any], regime: str) -> dict[str, RE.ExportBundle]:
    """The baseline families that describe ``regime``, read from their files."""
    out: dict[str, RE.ExportBundle] = {}
    for family, regimes in dict(cohort.get("families") or {}).items():
        entry = dict(regimes).get(regime)
        if not entry:
            continue
        out[family] = read_export_cached(
            require(entry, "path", f"cohort {cohort['name']} family {family}"),
            family=family,
            regime=regime,
            declared=entry.get("provenance"),
        )
    return out


def all_families(cohort: dict[str, Any]) -> dict[str, dict[str, RE.ExportBundle]]:
    return {
        str(rs["regime"]): cohort_families(cohort, str(rs["regime"])) for rs in cohort["regimes"]
    }


def resolved_dataset_version(dataset: str, declared: str | None) -> dict[str, Any]:
    """The dataset version actually opened, resolved through the repository.

    A ``null`` version in the manifest is not a pin: the run records the digest
    dload resolves, and ``--check`` fails if it moved.
    """
    from data_processing.streams import open_repository

    name, pinned = C.split_dataset(str(dataset))
    version = declared or pinned
    ds = open_repository().dataset(name, version)
    return dict(dataset=name, declared_version=version, resolved_version=str(ds.version))


def frozen_inputs(man: dict[str, Any], scorer: dict[str, Any]) -> dict[str, Any]:
    """Everything whose content must not move between ``--prepare`` and ``--check``."""
    exports: dict[str, Any] = {}
    datasets: dict[str, Any] = {}
    for cohort in man["cohorts"]:
        ds = resolved_dataset_version(str(cohort["dataset"]), cohort.get("version"))
        datasets[ds["dataset"]] = ds
        for family, regimes in dict(cohort.get("families") or {}).items():
            for regime, entry in dict(regimes).items():
                path = str(require(dict(entry), "path", f"family {family}"))
                exports[f"{cohort['name']}:{family}:{regime}"] = RE.artifact_digest(path) | dict(
                    declared_provenance=bool(dict(entry).get("provenance"))
                )
    cand = dict(dict(man["arms"]).get("candidate") or {})
    if cand.get("export"):
        exports["candidate"] = RE.artifact_digest(str(cand["export"]))
    return dict(
        manifest=dict(path=man["_path"], sha256=man["_digest"]),
        scorer=scorer,
        exports=exports,
        datasets=datasets,
        observation=dict(
            n_fft=RE.OBS_N_FFT, hop=RE.OBS_HOP, f_min=RE.OBS_F_MIN, f_max=RE.OBS_F_MAX, sr=RE.SR
        ),
        rule=(
            "the frozen-manifest promise covers the referenced artifacts: manifest text, scorer "
            "checkpoint, every export's content digest and every resolved dataset version"
        ),
    )


def verify_frozen_inputs(now: dict[str, Any], then: dict[str, Any]) -> None:
    """Fail if any referenced artifact moved since the calibration."""
    if now["scorer"]["sha256"] != then["scorer"]["sha256"]:
        die("the scorer checkpoint digest differs from the calibration's — refusing to mix them")
    for key, rec in then.get("exports", {}).items():
        if key == "candidate":
            continue  # the candidate is new by construction; it is pinned in its own record
        cur = now.get("exports", {}).get(key)
        if cur is None:
            die(f"the manifest no longer references the calibrated export {key!r}")
        if cur["sha256"] != rec["sha256"]:
            die(
                f"export {key!r} changed since --prepare ({rec['sha256'][:12]} -> "
                f"{cur['sha256'][:12]}): its supports may have moved, so the calibration is void"
            )
    for name, rec in then.get("datasets", {}).items():
        cur = now.get("datasets", {}).get(name)
        if cur is None or cur["resolved_version"] != rec["resolved_version"]:
            die(
                f"dataset {name} resolves to "
                f"{(cur or {}).get('resolved_version')} but the calibration used "
                f"{rec['resolved_version']}"
            )


def verify_protocol_fingerprint(man: dict[str, Any], cal: dict[str, Any]) -> str:
    """Return the verified protocol fingerprint, refusing protocol drift.

    A calibration produced before the protocol-fingerprint field carries its
    original manifest SHA and path; we re-derive the protocol fingerprint from
    that recorded manifest (after verifying it still hashes to the same value)
    and compare it to the current manifest's protocol fingerprint. New
    calibrations store the protocol fingerprint directly.
    """
    current = protocol_fingerprint(man)
    cal_manifest = dict(cal.get("manifest") or {})
    stored_protocol = cal_manifest.get("protocol_sha256")
    if stored_protocol is not None:
        if stored_protocol != current:
            die(
                f"protocol fingerprint {current[:12]} != calibration's "
                f"{stored_protocol[:12]} — gates, observations, cohorts, baseline families or "
                "seeds changed since the calibration was derived"
            )
        return current
    # Adopt a pre-fingerprint calibration: verify the recorded manifest still
    # hashes to its declared SHA, derive its protocol fingerprint without
    # re-rendering, and compare to the current manifest.
    orig_path = Path(cal_manifest.get("path", man["_path"]))
    orig_sha = cal_manifest.get("sha256")
    if orig_sha is None or not orig_path.is_file():
        die(
            "calibration has no protocol fingerprint and no verifiable original manifest; "
            "re-run --prepare"
        )
    raw = orig_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != orig_sha:
        die(
            f"recorded manifest {orig_path} no longer hashes to {orig_sha[:12]}; "
            "do not forge old provenance"
        )
    orig_man = json.loads(raw.decode())
    orig_man["_digest"] = orig_sha
    orig_man["_path"] = str(orig_path)
    derived = protocol_fingerprint(orig_man)
    if derived != current:
        die(
            f"derived protocol fingerprint {current[:12]} != calibration-era "
            f"{derived[:12]} — gates, observations, cohorts, baseline families or seeds changed"
        )
    return current


# ── arms ────────────────────────────────────────────────────────────────────


@dataclass
class ArmModel:
    """One arm's model: a legacy parameter set or a revised export."""

    name: str
    kind: str  # real | legacy | revised
    label: str = ""
    legacy: RE.ModelParams | None = None
    candidate: RE.CandidateExport | None = None
    spec: dict[str, Any] = field(default_factory=dict)

    def source(self) -> dict[str, Any]:
        if self.kind == "legacy" and self.legacy is not None:
            return dict(
                model_family="legacy",
                observation_law=RE.HISTORICAL_FORWARD_LABEL,
                aggregate=dict(self.legacy.source),
            )
        if self.kind == "revised" and self.candidate is not None:
            return dict(
                model_family=self.candidate.model_family,
                observation_law=(
                    "revised_phase.predict_spectrum(mode='prior'): the exact moving-window "
                    "kernel, raw telemetry plus the learned bias law, m=0, full prior kernel"
                ),
                candidate=self.candidate.provenance(),
                identified=self.candidate.identified,
            )
        return dict(model_family=self.kind, label=self.label)

    def render(
        self, reference_rps: np.ndarray, *, n_mics: int, seed: int
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
        """``(audio, physical_rps | None, diagnostics)`` at 16 kHz."""
        if self.kind == "legacy":
            assert self.legacy is not None
            physical = RE.to_renderer_units(self.legacy)
            audio = S2.render_from_export(
                dict(physical.params),
                reference_rps,
                sample_rate_work=RE.SAMPLE_RATE_WORK,
                n_mics=n_mics,
                seed=int(seed),
                normalize_rms=None,  # the evaluator is the physical-units path
            )
            return (
                np.asarray(audio, dtype=np.float64),
                None,
                dict(
                    normalize_rms=None,
                    render_units=physical.source["render_units"],
                    physical_rps="not emitted by the legacy renderer",
                ),
            )
        if self.kind == "revised":
            assert self.candidate is not None
            out = RP.render_revised(
                self.candidate.summary,
                reference_rps,
                sample_rate=RE.SR,
                sample_rate_work=RE.SAMPLE_RATE_WORK,
                n_mics=int(n_mics),
                seed=int(seed),
            )
            return (
                np.asarray(out.audio, dtype=np.float64),
                np.asarray(out.physical_rps, dtype=np.float64),
                dict(out.diagnostics),
            )
        raise AssertionError(f"arm {self.name}: kind {self.kind!r} does not render")

    def spectrum(self, clip: Clip, *, n_mics: int) -> np.ndarray | None:
        """``(M, N, F)`` predicted periodogram on the frozen observation grid."""
        if self.kind == "legacy" and self.legacy is not None:
            pg = RE.window_periodogram(clip)
            return RE.predicted_m(self.legacy, pg, n_mics=n_mics)
        if self.kind == "revised" and self.candidate is not None:
            m = np.asarray(
                RP.predict_spectrum(
                    self.candidate.summary, clip, n_fft=RE.OBS_N_FFT, hop=RE.OBS_HOP, mode="prior"
                ),
                dtype=np.float64,
            )
            if m.shape[0] < n_mics:
                die(
                    f"arm {self.name}: predict_spectrum returned {m.shape[0]} microphones, the "
                    f"frozen set needs {n_mics}"
                )
            return m[:n_mics]
        return None


def _resolve_candidate_export(
    spec: dict[str, Any], rig: str | None, *, context: str
) -> tuple[str, str | None]:
    """Return ``(export_path, fit_manifest_sha256)`` for a candidate arm.

    Supports a single export (legacy/smoke) or a ``per_rig`` mapping. The
    per-rig form is required for a multi-rig evaluation so that each rig's
    export is pinned and validated against that rig.
    """
    per_rig = spec.get("per_rig")
    if per_rig is not None:
        if rig is None:
            die(f"{context}: per_rig candidate mapping requires a cohort rig")
        entry = dict(per_rig).get(rig)
        if entry is None:
            die(f"{context}: no candidate export declared for rig {rig!r}")
        return str(require(entry, "export", f"{context} rig {rig}")), entry.get("fit_manifest_sha256")
    return str(require(spec, "export", context)), spec.get("fit_manifest_sha256")


def arm_model(
    name: str, spec: dict[str, Any], *, regime: str, recording: str, rig: str | None = None
) -> ArmModel:
    """Resolve one arm's model by DISPATCHING on the export's family."""
    kind = str(require(spec, "kind", f"arm {name}"))
    if kind == "real":
        return ArmModel(name, "real", label="real audio", spec=spec)
    if kind == "revised_export":
        export_path, _ = _resolve_candidate_export(spec, rig, context=f"arm {name}")
        cand = RE.read_candidate_export(export_path)
        if rig is not None and cand.rig_id != rig:
            die(
                f"arm {name}: candidate export rig_id is {cand.rig_id!r}, "
                f"cohort rig is {rig!r}"
            )
        return ArmModel(
            name,
            "revised",
            label=str(spec.get("label", cand.model_family)),
            candidate=cand,
            spec=spec,
        )
    if kind == "export_render":
        path = spec.get("export")
        if path is None:
            raise AssertionError("export_render arms resolve their export through baseline_params")
        bundle = read_export_cached(
            str(path), family="candidate", regime=regime, declared=spec.get("provenance")
        )
        ids = [cid for cid, _ in bundle.clips_of(recording)]
        if not ids:
            if not bool(spec.get("extrapolated")) or not spec.get("label"):
                die(
                    f"arm {name}: {bundle.path} does not name {recording!r}; declare "
                    '"extrapolated": true with a label to use its aggregate instead'
                )
            ids = list(bundle.clip_ids)
        mp = RE.aggregate_nuisance(
            bundle,
            ids,
            label=str(spec.get("label", name)),
            extrapolated=bool(spec.get("extrapolated")),
        )
        return ArmModel(name, "legacy", label=mp.source["label"], legacy=mp, spec=spec)
    die(f"arm {name}: unknown kind {kind!r}")
    raise AssertionError


def baseline_params(
    cohort: dict[str, Any],
    regime: str,
    recording: str,
    *,
    family: str,
    bundles_by_regime: dict[str, dict[str, RE.ExportBundle]],
) -> RE.ModelParams:
    """The baseline parameter set for one (regime, recording), by the frozen map.

    ``match: identity`` requires the export to NAME the recording and uses that
    recording's own clips. ``match: aggregate`` is a labelled cross-recording
    extrapolation over every clip of the export — the only route for Michael's
    held-out FLY124, and for its ramp, which has no fitted baseline at all.
    Both routes are explicit; there is no positional or modulo route.
    """
    rule = dict(dict(cohort.get("baseline_map") or {}).get(regime) or {})
    if not rule:
        die(f"cohort {cohort['name']}: no baseline_map entry for regime {regime!r}")
    src_regime = str(rule.get("from_regime", regime))
    bundle = (bundles_by_regime.get(src_regime) or {}).get(family)
    if bundle is None:
        die(
            f"cohort {cohort['name']}: family {family!r} has no export for regime "
            f"{src_regime!r}, so regime {regime!r} cannot be given a baseline"
        )
    assert bundle is not None
    match = str(rule.get("match", "identity"))
    if match == "identity":
        pairs = bundle.clips_of(recording)
        if not pairs:
            die(
                f"{bundle.path}: does not name {recording!r} (it names "
                f"{list(bundle.recordings)}) — identity pairing failed and there is no "
                "positional fallback"
            )
        clip_ids, extrapolated = [cid for cid, _ in pairs], False
    elif match == "aggregate":
        if not bool(rule.get("extrapolated")) or not rule.get("label"):
            die(
                f"cohort {cohort['name']} regime {regime!r}: match 'aggregate' is a "
                'cross-recording extrapolation and must carry "extrapolated": true and a label'
            )
        clip_ids, extrapolated = list(bundle.clip_ids), True
    else:
        die(f"cohort {cohort['name']} regime {regime!r}: unknown match {match!r}")
    mp = RE.aggregate_nuisance(
        bundle,
        clip_ids,
        label=str(rule.get("label", f"{family}:{src_regime}:{match}")),
        extrapolated=extrapolated,
    )
    mp.source.update(baseline_rule=rule, for_recording=recording, for_regime=regime)
    return mp


# ── window resolution ───────────────────────────────────────────────────────


def resolve_windows(
    cohort: dict[str, Any], regime_spec: dict[str, Any], calibration: list[RE.Window]
) -> tuple[list[RE.Window], list[dict[str, Any]]]:
    """Held-out windows plus the support report for every cohort recording.

    The regime test reads the SAME raw telemetry key the scoring uses, so no
    refined/posterior label enters the held-out path at all — not even through
    window selection.
    """
    regime = str(require(regime_spec, "regime", f"cohort {cohort['name']}"))
    ev = dict(require(regime_spec, "evaluation", f"cohort {cohort['name']} regime {regime}"))
    mode = str(ev.get("mode", "auto"))
    if mode == "explicit":
        declared = [
            RE.Window(
                str(w["recording"]),
                float(w["start_s"]),
                float(w["duration_s"]),
                regime=regime,
                role="evaluation",
            )
            for w in ev.get("windows", [])
        ]
        kept, reports = RE.check_explicit_windows(declared, regime=regime, calibration=calibration)
        return kept, [r.as_dict() | dict(mode="explicit") for r in reports]
    dataset, pinned = C.split_dataset(str(require(cohort, "dataset", f"cohort {cohort['name']}")))
    version = cohort.get("version") or pinned
    key = RE.assert_raw_reference(str(require(cohort, "scoring_rps_key", "cohort")))
    kept: list[RE.Window] = []
    reports: list[dict[str, Any]] = []
    for rid in list(require(cohort, "recordings", f"cohort {cohort['name']}")):
        rec = C.load_recording(dataset, rid, version, key)
        if mode == "support":
            # a regime shorter than one window: the window is CONTEXT, the
            # scored material stays the pre-registered interval inside it
            got, report = RE.resolve_support_windows(
                rec,
                regime=regime,
                calibration=calibration,
                context_seconds=float(require(ev, "context_seconds", "evaluation")),
                min_support_seconds=float(require(ev, "min_support_seconds", "evaluation")),
                max_windows=int(ev.get("max_windows", 1)),
                min_rps=regime_spec.get("min_rps"),
                max_rps=regime_spec.get("max_rps"),
            )
        elif mode == "auto":
            got, report = RE.resolve_evaluation_windows(
                rec,
                regime=regime,
                calibration=calibration,
                min_seconds=float(require(ev, "min_seconds", "evaluation")),
                stride_s=ev.get("stride_s"),
                max_windows=int(ev.get("max_windows", 1)),
                min_rps=regime_spec.get("min_rps"),
                max_rps=regime_spec.get("max_rps"),
            )
        else:
            die(f"cohort {cohort['name']} regime {regime}: unknown evaluation mode {mode!r}")
            raise AssertionError
        kept.extend(got)
        reports.append(report.as_dict() | dict(mode=mode, selection_rps_key=key))
    return kept, reports


# ── measurement ─────────────────────────────────────────────────────────────


class Tracker:
    """The frozen scorer, loaded once, with its checkpoint digest recorded."""

    def __init__(self, scorer: dict[str, Any]) -> None:
        import zoo
        from metrics.salience_layers import LayerPeakRPSMetric

        name, ckpt = str(scorer["experiment"]), str(scorer.get("ckpt", "best"))
        self.record = checkpoint_record(name, ckpt)
        self.fm = zoo.load(name, ckpt=ckpt)
        self.metric = LayerPeakRPSMetric()
        self.score = probe_helpers().score

    def pit(
        self,
        audio: np.ndarray,
        reference: np.ndarray,
        *,
        mics: list[int],
        expected_samples: int,
        support: RE.RegimeSupport | None,
    ) -> dict[str, Any]:
        return RE.pit_mae(
            self.score,
            self.fm,
            self.metric,
            audio,
            reference,
            sr=RE.SR,
            mics=mics,
            expected_samples=expected_samples,
            support=support,
        )


def scored_ltas(
    real: np.ndarray, arm: np.ndarray, support: RE.RegimeSupport
) -> dict[str, Any] | None:
    """Absolute-level LTAS on the LONGEST contiguous scored span, or ``None``.

    The LTAS needs ONE complete 8192-point Welch window inside the scored
    support; with the half-hop geometry that is ``>= LTAS_N`` samples, not
    ``2 * LTAS_N``. Material shorter than one window has no LTAS, which is
    reported rather than papered over with the surrounding context.
    """
    spans: list[tuple[int, int]] = []
    m = np.asarray(support.sample_mask, dtype=bool)
    padded = np.concatenate(([False], m, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    for a, b in zip(edges[0::2], edges[1::2], strict=True):
        spans.append((int(a), int(b)))
    if not spans:
        return None
    a, b = max(spans, key=lambda s: s[1] - s[0])
    if (b - a) < LTAS_N:
        return None
    out = RE.ltas_deviation_db(real[:, a:b], arm[:, a:b])
    out["scored_span_samples"] = [a, b]
    out["scored_span_seconds"] = float((b - a) / support.sr)
    return out


def measure_window(
    window: RE.Window,
    *,
    cohort: dict[str, Any],
    regime_spec: dict[str, Any],
    tracker: Tracker,
    models: dict[str, ArmModel],
    seed: int,
    n_mics: int | None,
    want_spectrum: bool,
) -> dict[str, Any]:
    """Every metric of one window for every arm, paired on the window identity."""
    dataset = str(cohort["dataset"])
    key = RE.assert_raw_reference(str(require(cohort, "scoring_rps_key", "cohort")))
    regime = str(regime_spec["regime"])
    clip = RE.load_window(
        window,
        dataset=dataset,
        version=cohort.get("version"),
        channels=cohort.get("channels"),
        rps_key=key,
    )
    real_all = np.asarray(clip.audio, dtype=np.float64)
    mics = list(range(int(n_mics or real_all.shape[0])))
    real = real_all[: len(mics)]
    reference = np.asarray(clip.rps, dtype=np.float64)
    n_samples = int(real.shape[-1])
    support = RE.regime_support(
        window,
        reference,
        regime=regime,
        min_rps=regime_spec.get("min_rps"),
        max_rps=regime_spec.get("max_rps"),
        sr=clip.sr,
    )
    scored_clip = Clip(clip.clip_id, clip.group, real.astype(np.float32), reference, clip.sr)
    pg = RE.window_periodogram(scored_clip)
    band = RE.observation_band(pg.freqs)
    frame_keep = (
        np.zeros(0, dtype=bool)
        if pg.times.size == 0
        else support.sample_mask[
            np.clip((np.asarray(pg.times) * clip.sr).astype(int), 0, n_samples - 1)
        ]
    )
    row: dict[str, Any] = dict(
        window=window.as_dict(),
        regime=regime,
        seed=int(seed),
        scoring_reference=dict(
            rps_key=key,
            note="same raw telemetry array for every arm; reference agreement, not ground truth",
            mean_rps=float(reference.mean()),
            per_rotor_mean_rps=[float(v) for v in reference.mean(axis=1)],
        ),
        mics=mics,
        n_samples=n_samples,
        n_frames_total=int(pg.times.size),
        n_frames_scored=int(frame_keep.sum()),
        band_bins=int(band.sum()),
        support=support.as_dict(),
        arms={},
    )
    if support.n_scored == 0:
        row["unavailable"] = (
            f"the pre-registered {regime} support selects no sample of this window: it is "
            "context only and nothing here is scored"
        )
        return row
    if pg.times.size == 0 or not bool(frame_keep.any()):
        row["spectral_unavailable"] = (
            f"{support.scored_seconds:.3f} s of scored {regime} support carries no "
            f"{RE.OBS_N_FFT}-point analysis frame centre: PIT only for this window"
        )
    for name, model in models.items():
        spec = model.spec
        kind = str(spec["kind"])
        physical: np.ndarray | None = None
        diag: dict[str, Any] = {}
        if kind == "real":
            audio = real
        else:
            audio, physical, diag = model.render(reference, n_mics=len(mics), seed=int(seed))
            audio = audio[: len(mics)]
        entry: dict[str, Any] = dict(kind=kind, seed=int(seed), label=model.label, render=diag)
        t0 = time.time()
        entry["pit"] = tracker.pit(
            audio,
            reference,
            mics=mics,
            expected_samples=n_samples,
            support=support,
        )
        entry["pit"]["seconds"] = time.time() - t0
        if name != "real":
            entry["ltas"] = scored_ltas(real, audio, support)
            if entry["ltas"] is None:
                entry["ltas_unavailable"] = (
                    f"the longest contiguous scored span is shorter than {LTAS_N} samples, "
                    "so no absolute LTAS is computed for this window"
                )
        if physical is not None:
            sm = support.sample_mask[: physical.shape[-1]]
            ref_cut = reference[:, : physical.shape[-1]]
            entry["physical_shaft_mae_diagnostic"] = dict(
                value=float(np.abs(physical[:, sm] - ref_cut[:, sm]).mean()),
                per_rotor=[float(v) for v in np.abs(physical[:, sm] - ref_cut[:, sm]).mean(axis=1)],
                role=(
                    "DIAGNOSTIC ONLY: realized physical shaft rev/s against the shared raw "
                    "telemetry reference. Never a gate, never the primary target."
                ),
                definition=diag.get("physical_rps_definition"),
                eps_excluded=diag.get("eps_in_physical_rps"),
            )
        if want_spectrum and str(spec["kind"]) != "real" and bool(frame_keep.any()):
            m = model.spectrum(scored_clip, n_mics=len(mics))
            if m is not None:
                power = np.asarray(pg.power, dtype=np.float64)[: len(mics)]
                if m.shape != power.shape:
                    die(
                        f"arm {name}: predicted spectrum {m.shape} does not match the frozen "
                        f"observation {power.shape}"
                    )
                nll, cells = RE.marginal_frame_nll(power, m, band)
                times = RE.frame_times_on_clock(window, pg)
                entry["composite_frames"] = dict(
                    frame_nll=[float(v) for v in nll[frame_keep]],
                    frame_times=[float(v) for v in times[frame_keep]],
                    cells_per_frame=cells,
                    n_fft=RE.OBS_N_FFT,
                    hop=RE.OBS_HOP,
                    scored_of_total=[int(frame_keep.sum()), int(frame_keep.size)],
                )
                entry["model_source"] = model.source()
        row["arms"][name] = entry
    return row


def frame_scores(rows: list[dict[str, Any]], arm: str) -> list[RE.FrameScore]:
    """The composite inputs of one arm, carrying the frames they came from."""
    out: list[RE.FrameScore] = []
    for row in rows:
        cf = (row["arms"].get(arm) or {}).get("composite_frames")
        if not cf:
            continue
        w = row["window"]
        out.append(
            RE.FrameScore(
                window=RE.Window(
                    w["recording"], float(w["start_s"]), float(w["duration_s"]), regime=w["regime"]
                ),
                frame_times=np.asarray(cf["frame_times"], dtype=np.float64),
                frame_nll=np.asarray(cf["frame_nll"], dtype=np.float64),
                n_cells_per_frame=int(cf["cells_per_frame"]),
                n_fft=int(cf.get("n_fft", RE.OBS_N_FFT)),
                hop=int(cf.get("hop", RE.OBS_HOP)),
            )
        )
    return out


def _values(rows: list[dict[str, Any]], arm: str, metric: str) -> list[tuple[str, str, float]]:
    out: list[tuple[str, str, float]] = []
    for row in rows:
        entry = row["arms"].get(arm) or {}
        if metric == "pit" and entry.get("pit"):
            out.append(
                (row["window"]["recording"], row["window"]["key"], float(entry["pit"]["mae"]))
            )
        if metric == "ltas" and entry.get("ltas"):
            out.append(
                (
                    row["window"]["recording"],
                    row["window"]["key"],
                    float(entry["ltas"]["mean_abs_db"]),
                )
            )
    return out


def per_recording(rows: list[dict[str, Any]], arm: str, metric: str) -> dict[str, float]:
    """Cluster values: the mean over that recording's windows and render seeds."""
    acc: dict[str, list[float]] = {}
    for rid, _key, v in _values(rows, arm, metric):
        acc.setdefault(rid, []).append(v)
    return {r: float(np.mean(v)) for r, v in sorted(acc.items())}


def per_window(rows: list[dict[str, Any]], arm: str, metric: str) -> dict[str, float]:
    """Block values: the mean over render seeds of one window identity."""
    acc: dict[str, list[float]] = {}
    for _rid, key, v in _values(rows, arm, metric):
        acc.setdefault(key, []).append(v)
    return {k: float(np.mean(v)) for k, v in sorted(acc.items())}


def composite_per_window(rows: list[dict[str, Any]], arm: str) -> dict[str, float]:
    """Per-block composite risk — the pairing unit of the Michael gates."""
    out: dict[str, float] = {}
    for row in rows:
        items = frame_scores([row], arm)
        if not items:
            continue
        comp = RE.composite_score(items)
        if comp["score"] is not None:
            out[row["window"]["key"]] = float(comp["score"])
    return out


# ── modes ───────────────────────────────────────────────────────────────────


def cohort_plan(cohort: dict[str, Any]) -> dict[str, Any]:
    """Cohort, supports and family eligibility — no scoring, no rendering."""
    name = str(require(cohort, "name", "cohort"))
    plan: dict[str, Any] = dict(
        name=name,
        rig=cohort.get("rig"),
        role=cohort.get("role"),
        gate=cohort.get("gate"),
        dataset=cohort.get("dataset"),
        recordings=list(cohort.get("recordings") or []),
        scoring_rps_key=RE.assert_raw_reference(str(require(cohort, "scoring_rps_key", name))),
        regimes={},
    )
    for regime_spec in list(require(cohort, "regimes", name)):
        regime = str(regime_spec["regime"])
        bundles = cohort_families(cohort, regime)
        calibration = [w for b in bundles.values() for w in b.calibration_windows()]
        windows, reports = resolve_windows(cohort, regime_spec, calibration)
        plan["regimes"][regime] = dict(
            families={f: b.provenance() for f, b in bundles.items()},
            coverage={
                f: c.as_dict()
                for f, c in RE.family_coverage(
                    bundles, list(cohort.get("recordings") or [])
                ).items()
            },
            baseline_rule=dict(cohort.get("baseline_map") or {}).get(regime),
            regime_band=dict(
                min_rps=regime_spec.get("min_rps"), max_rps=regime_spec.get("max_rps")
            ),
            calibration_windows=[w.as_dict() for w in calibration],
            evaluation_windows=[w.as_dict() for w in windows],
            support_reports=reports,
            insufficient=[r["recording"] for r in reports if not r["sufficient"]],
        )
    return plan


def calibration_temperature(
    cohort: dict[str, Any],
    calibration: list[RE.Window],
    mp_for_window: dict[str, RE.ModelParams],
    *,
    tracker: Tracker,
    n_mics: int | None,
    master_seed: int,
    B: int,
) -> dict[str, Any]:
    """Compute the frozen scalar composite temperature ``T = J / H`` for one rig.

    Renders ``B`` predictive draws of the frozen legacy baseline on the given
    calibration supports (which may span multiple regimes), computes the
    gain-direction score ``U_b = sum_i a_i (1 - I_b / M_i)`` on the actual
    unnormalized composite-risk weights ``a_i = hop / n_fft`` (duplicate-split),
    and returns :func:`revised_eval.composite_temperature`.
    """
    if B < 2:
        die("composite_temperature.B must be at least 2")
    rng = np.random.default_rng(int(master_seed))
    draw_seeds = [int(s) for s in rng.integers(0, 2**31, size=B)]
    key = RE.assert_raw_reference(str(cohort["scoring_rps_key"]))
    dataset = str(cohort["dataset"])
    version = cohort.get("version")
    channels = cohort.get("channels")
    items: list[RE.FrameScore] = []
    draw_terms: list[list[np.ndarray]] = [[] for _ in range(B)]
    for w in calibration:
        mp = mp_for_window.get(w.key)
        if mp is None:
            die(f"temperature calibration: no baseline params for {w.key}")
        clip = RE.load_window(w, dataset=dataset, version=version, channels=channels, rps_key=key)
        mics = int(n_mics or clip.audio.shape[0])
        reference = np.asarray(clip.rps, dtype=np.float64)
        pg = RE.window_periodogram(clip)
        band = RE.observation_band(pg.freqs)
        M = RE.predicted_m(mp, pg, n_mics=mics)
        power = np.asarray(pg.power, dtype=np.float64)[:mics]
        nll, cells = RE.marginal_frame_nll(power, M, band)
        items.append(
            RE.FrameScore(w, RE.frame_times_on_clock(w, pg), nll, cells, RE.OBS_N_FFT, RE.OBS_HOP)
        )
        physical = RE.to_renderer_units(mp)
        for b_idx, seed in enumerate(draw_seeds):
            audio = S2.render_from_export(
                dict(physical.params),
                reference,
                sample_rate_work=RE.SAMPLE_RATE_WORK,
                n_mics=mics,
                seed=seed,
                normalize_rms=None,
            )
            audio = np.asarray(audio, dtype=np.float64)[:mics]
            syn = Clip(
                f"{w.key}_draw{seed}",
                "synthetic",
                audio.astype(np.float32),
                reference[:, : audio.shape[-1]],
                RE.SR,
            )
            syn_pg = RE.window_periodogram(syn)
            syn_power = np.asarray(syn_pg.power, dtype=np.float64)[:mics]
            if syn_power.shape != M.shape:
                die(
                    f"temperature draw for {w.key} seed {seed}: spectrum shape {syn_power.shape} "
                    f"does not match expected {M.shape}"
                )
            gain, _ = RE.gain_frame_terms(syn_power, M, band)
            draw_terms[b_idx].append(gain)
    total = RE.unnormalized_gain_total(items, draw_terms)
    return RE.composite_temperature(
        total["u_by_draw"],
        exposure_h=total["exposure_h"],
        master_seed=master_seed,
        clips=[w.key for w in calibration],
        rig=str(cohort.get("rig", cohort["name"])),
    )


def run_plan(man: dict[str, Any], out: Path, *, provenance: dict[str, Any] | None = None) -> int:
    payload = dict(
        mode="plan",
        schema=RE.SCHEMA,
        manifest=dict(path=man["_path"], sha256=man["_digest"]),
        import_provenance=provenance,
        observation=dict(
            n_fft=RE.OBS_N_FFT, hop=RE.OBS_HOP, f_min=RE.OBS_F_MIN, f_max=RE.OBS_F_MAX, sr=RE.SR
        ),
        gates=man["gates"],
        caveat=RE.ADAPTIVE_SELECTION_CAVEAT,
        cohorts=[cohort_plan(c) for c in man["cohorts"]],
    )
    RE.write_json(out / "plan.json", payload)
    for c in payload["cohorts"]:
        for regime, d in c["regimes"].items():
            print(
                f"[{c['name']}/{regime}] calibration {len(d['calibration_windows'])} windows, "
                f"held-out {len(d['evaluation_windows'])}, eligible families "
                f"{[f for f, v in d['coverage'].items() if v['eligible']]}"
            )
            for r in d["support_reports"]:
                if not r["sufficient"]:
                    print(f"   INSUFFICIENT {r['recording']}: {r['note']}")
    print(f"wrote {out / 'plan.json'}")
    return 0


def calibration_scores(
    cohort: dict[str, Any],
    regime_spec: dict[str, Any],
    bundles: dict[str, RE.ExportBundle],
    *,
    n_mics: int | None = None,
) -> dict[str, Any]:
    """Family comparison on CALIBRATION supports only, then one global choice.

    The composite risk of each family's own fit windows, identity-matched per
    recording. Held-out supports never take part in choosing a family, and the
    choice is global per rig/regime — never per window.

    ``family_selection.basis_recordings`` says which recordings the comparison
    reads: for DREGON that is the cohort itself, for Michael's it is the FIT
    recording (FLY125), because no export names the held-out FLY124 and its
    baseline is a declared extrapolation.
    """
    regime = str(regime_spec["regime"])
    fs = dict(cohort.get("family_selection") or {})
    basis = [str(r) for r in (fs.get("basis_recordings") or cohort["recordings"])]
    coverage = RE.family_coverage(bundles, list(cohort["recordings"]))
    scores: dict[str, float] = {}
    detail: dict[str, Any] = {}
    key = RE.assert_raw_reference(str(cohort["scoring_rps_key"]))
    for family, bundle in bundles.items():
        items: list[RE.FrameScore] = []
        for w in bundle.calibration_windows():
            if w.recording not in basis:
                continue
            mp = RE.aggregate_nuisance(
                bundle, [cid for cid, _ in bundle.clips_of(w.recording)], label=f"{family}:cal"
            )
            clip = RE.load_window(
                w,
                dataset=str(bundle.dataset or cohort["dataset"]),
                version=cohort.get("version"),
                channels=cohort.get("channels"),
                rps_key=key,
            )
            mics = int(n_mics or clip.audio.shape[0])
            pg = RE.window_periodogram(clip)
            band = RE.observation_band(pg.freqs)
            m = RE.predicted_m(mp, pg, n_mics=mics)
            power = np.asarray(pg.power, dtype=np.float64)[:mics]
            nll, cells = RE.marginal_frame_nll(power, m, band)
            items.append(
                RE.FrameScore(
                    w, RE.frame_times_on_clock(w, pg), nll, cells, RE.OBS_N_FFT, RE.OBS_HOP
                )
            )
            print(f"   [family {family}] {w.key}: {len(nll)} frames", flush=True)
        comp = RE.composite_score(items)
        detail[family] = comp
        if comp["score"] is not None:
            scores[family] = float(comp["score"])
    return dict(
        regime=regime,
        basis_recordings=basis,
        selection=RE.select_family(
            scores, coverage, require_coverage=bool(fs.get("require_identity_coverage", True))
        ),
        per_family=detail,
        observation_law=RE.HISTORICAL_FORWARD_LABEL,
        basis="calibration (fit) supports only — held-out supports never choose the family",
    )


def measure_arm_set(
    cohort: dict[str, Any],
    regime_spec: dict[str, Any],
    windows: list[RE.Window],
    *,
    tracker: Tracker,
    arms: dict[str, dict[str, Any]],
    model_for: Any,
    seeds: list[int],
    n_mics: int | None,
    tag: str,
) -> list[dict[str, Any]]:
    """Measure ``arms`` on every window, over every FROZEN render seed."""
    regime = str(regime_spec["regime"])
    rows: list[dict[str, Any]] = []
    for w in windows:
        models = {name: model_for(name, spec, w) for name, spec in arms.items()}
        static = {n for n, s in arms.items() if str(s["kind"]) == "real"}
        for j, seed in enumerate(seeds):
            active = (
                dict(models) if j == 0 else {n: m for n, m in models.items() if n not in static}
            )
            if not active:
                continue
            row = measure_window(
                w,
                cohort=cohort,
                regime_spec=regime_spec,
                tracker=tracker,
                models=active,
                seed=int(seed),
                n_mics=n_mics,
                want_spectrum=(j == 0),
            )
            rows.append(row)
            summary = "  ".join(
                f"{n}={row['arms'][n]['pit']['mae']:.4f}" for n in row.get("arms", {})
            )
            note = row.get("unavailable") or row.get("spectral_unavailable") or ""
            print(f"[{tag}/{regime}] {w.key} seed {seed}: {summary} {note}", flush=True)
    return rows


def run_prepare(
    man: dict[str, Any], out: Path, *, n_mics: int | None, provenance: dict[str, Any] | None = None
) -> int:
    tracker = Tracker(dict(man["scorer"]))
    nullvar = dict(man["null_variation"])
    seeds = [int(s) for s in require(nullvar, "seeds", "null_variation")]
    if len(seeds) < 2:
        die("null_variation.seeds must hold at least two baseline render seeds")
    baseline_spec = dict(require(dict(man["arms"]), "baseline", "arms"))
    record: dict[str, Any] = dict(
        mode="prepare",
        schema=RE.SCHEMA,
        manifest=dict(
            path=man["_path"],
            sha256=man["_digest"],
            protocol_sha256=protocol_fingerprint(man),
        ),
        import_provenance=provenance,
        scorer=tracker.record,
        frozen_inputs=frozen_inputs(man, tracker.record),
        gates=man["gates"],
        null_variation_rule=nullvar,
        render_units=dict(
            normalize_rms=None,
            rate_factor_db=RE.RENDER_RATE_FACTOR_DB,
            note="the declared conversion of revised_eval.to_renderer_units, applied to every "
            "legacy arm; legacy exported absolute levels are stale without it",
        ),
        caveat=RE.ADAPTIVE_SELECTION_CAVEAT,
        cohorts={},
    )
    for cohort in man["cohorts"]:
        name = str(cohort["name"])
        bundles_by_regime = all_families(cohort)
        cres: dict[str, Any] = dict(plan=cohort_plan(cohort), regimes={})
        temp_config = dict(man.get("composite_temperature") or {})
        temp_master = temp_config.get("master_seed")
        temp_B = temp_config.get("B")
        all_calibration: list[RE.Window] = []
        all_mp_for_window: dict[str, RE.ModelParams] = {}
        for regime_spec in cohort["regimes"]:
            regime = str(regime_spec["regime"])
            bundles = bundles_by_regime[regime]
            sel = (
                calibration_scores(cohort, regime_spec, bundles, n_mics=n_mics)
                if bundles and bool(regime_spec.get("select_family", True))
                else dict(regime=regime, selection=dict(chosen=None), per_family={})
            )
            family = sel["selection"].get("chosen") or str(
                require(dict(regime_spec), "family", f"{name}:{regime}")
            )
            calibration = [w for b in bundles.values() for w in b.calibration_windows()]
            windows, reports = resolve_windows(cohort, regime_spec, calibration)

            for w in calibration:
                if w.key not in all_mp_for_window:
                    all_mp_for_window[w.key] = baseline_params(
                        cohort,
                        regime,
                        w.recording,
                        family=family,
                        bundles_by_regime=bundles_by_regime,
                    )
                    all_calibration.append(w)

            def model_for(
                arm: str,
                spec: dict[str, Any],
                w: RE.Window,
                _r: str = regime,
                _f: str = family,
                _c: dict[str, Any] = cohort,
                _b: dict[str, dict[str, RE.ExportBundle]] = bundles_by_regime,
            ) -> ArmModel:
                if arm == "real":
                    return ArmModel("real", "real", label="real audio", spec=spec)
                mp = baseline_params(_c, _r, w.recording, family=_f, bundles_by_regime=_b)
                return ArmModel(arm, "legacy", label=mp.source["label"], legacy=mp, spec=spec)

            rows = measure_arm_set(
                cohort,
                regime_spec,
                windows,
                tracker=tracker,
                arms={"real": dict(kind="real"), "baseline": baseline_spec},
                model_for=model_for,
                seeds=seeds,
                n_mics=n_mics,
                tag=name,
            )
            cres["regimes"][regime] = dict(
                family_selection=sel,
                frozen_family=family,
                support_reports=reports,
                windows=[w.as_dict() for w in windows],
                measurements=rows,
            )
        temp_record: dict[str, Any] | None = None
        if temp_master is not None and temp_B is not None and all_calibration:
            temp_record = calibration_temperature(
                cohort,
                all_calibration,
                all_mp_for_window,
                tracker=tracker,
                n_mics=n_mics,
                master_seed=int(temp_master),
                B=int(temp_B),
            )
        cres["composite_temperature"] = temp_record
        record["cohorts"][name] = cres
    record["calibration"] = summarize_prepare(record, man)
    RE.write_json(out / "calibration.json", record)
    print(json.dumps(RE.json_ready(record["calibration"]), indent=1))
    print(f"wrote {out / 'calibration.json'}")
    return 0


def _abs_delta(a: dict[str, float], b: dict[str, float]) -> list[float]:
    return [abs(a[k] - b[k]) for k in sorted(set(a) & set(b))]


def summarize_prepare(record: dict[str, Any], man: dict[str, Any]) -> dict[str, Any]:
    """Gap calibration and the null-variation tolerances, per cohort and regime.

    Null variation is computed at BOTH pairing units, because the two rigs pair
    differently: DREGON clusters whole recordings, Michael's pairs blocks
    conditional on FLY124. A cohort with one recording therefore still gets a
    usable tolerance — from its own blocks — and if even that is unavailable the
    value stays ``None`` and the gate that needs it FAILS.
    """
    alpha = float(man["gates"]["alpha"])
    seed = int(man["gates"]["bootstrap_seed"])
    out: dict[str, Any] = dict(
        alpha=alpha,
        rule=(
            "null variation from the frozen baseline render seeds only: split the seeds into two "
            "disjoint halves, pair the halves per recording AND per block, and take the "
            "one-sided upper bound of the clustered |delta| as the tolerance. The composite risk "
            "has no render randomness (it is a function of the export and the real clip), so its "
            "null variation is exactly zero and its tolerance stays at the manifest value."
        ),
        caveat=RE.ADAPTIVE_SELECTION_CAVEAT,
        cohorts={},
    )
    for name, cres in record["cohorts"].items():
        per_regime: dict[str, Any] = {}
        for regime, rres in cres["regimes"].items():
            rows = rres["measurements"]
            seeds = sorted({int(r["seed"]) for r in rows if "baseline" in r.get("arms", {})})
            half_a, half_b = seeds[: len(seeds) // 2], seeds[len(seeds) // 2 :]
            real = per_recording(rows, "real", "pit")
            base = per_recording(rows, "baseline", "pit")
            gap = {r: base[r] - real[r] for r in sorted(set(real) & set(base))}
            gi = RE.cluster_interval(list(gap.values()), alpha=alpha, seed=seed)
            in_a = [r for r in rows if int(r["seed"]) in half_a]
            in_b = [r for r in rows if int(r["seed"]) in half_b]
            rec_pit = RE.cluster_interval(
                _abs_delta(
                    per_recording(in_a, "baseline", "pit"), per_recording(in_b, "baseline", "pit")
                ),
                alpha=alpha,
                seed=seed,
            )
            rec_ltas = RE.cluster_interval(
                _abs_delta(
                    per_recording(in_a, "baseline", "ltas"),
                    per_recording(in_b, "baseline", "ltas"),
                ),
                alpha=alpha,
                seed=seed,
            )
            blk_pit = RE.cluster_interval(
                _abs_delta(
                    per_window(in_a, "baseline", "pit"), per_window(in_b, "baseline", "pit")
                ),
                alpha=alpha,
                seed=seed,
            )
            blk_ltas = RE.cluster_interval(
                _abs_delta(
                    per_window(in_a, "baseline", "ltas"), per_window(in_b, "baseline", "ltas")
                ),
                alpha=alpha,
                seed=seed,
            )
            comp = RE.composite_score(frame_scores(rows, "baseline"))
            per_regime[regime] = dict(
                frozen_family=rres["frozen_family"],
                real_pit=real,
                baseline_pit=base,
                baseline_pit_by_window=per_window(rows, "baseline", "pit"),
                real_pit_by_window=per_window(rows, "real", "pit"),
                gap=gap,
                gap_interval=gi.as_dict(),
                gap_positive=bool(gi.lower is not None and gi.lower > 0.0),
                baseline_ltas_abs_db=per_recording(rows, "baseline", "ltas"),
                baseline_ltas_abs_db_by_window=per_window(rows, "baseline", "ltas"),
                baseline_composite=comp,
                baseline_composite_per_recording={
                    r: v["score"] for r, v in comp["per_recording"].items()
                },
                baseline_composite_by_window=composite_per_window(rows, "baseline"),
                support_reports=rres["support_reports"],
                observation_law=RE.HISTORICAL_FORWARD_LABEL,
                null_variation=dict(
                    seed_halves=[half_a, half_b],
                    recording_level=dict(
                        pit_tolerance=rec_pit.upper,
                        pit_interval=rec_pit.as_dict(),
                        ltas_tolerance_db=rec_ltas.upper,
                        ltas_interval=rec_ltas.as_dict(),
                    ),
                    block_level=dict(
                        pit_tolerance=blk_pit.upper,
                        pit_interval=blk_pit.as_dict(),
                        ltas_tolerance_db=blk_ltas.upper,
                        ltas_interval=blk_ltas.as_dict(),
                        conditioning="block variation within this cohort's own recording(s)",
                    ),
                    composite_tolerance=float(man["gates"]["composite_tolerance"]),
                    composite_note="deterministic given the export and the real clip",
                ),
                escalation=(
                    ""
                    if (
                        str(cres["plan"].get("gate") or "") != "dregon_pit"
                        or (gi.lower is not None and gi.lower > 0.0)
                    )
                    else "ESCALATE: the real-vs-synthetic gap is not distinguishable from zero at "
                    "this clustering — the DREGON target arithmetic is undefined until Main rules"
                ),
            )
        # Block variation CONDITIONAL ON THE COHORT'S RECORDING, pooled over
        # every held-out block of every regime: that is what "block variation
        # conditional on FLY124" means, and a per-regime pool of one window
        # would yield no tolerance at all.
        all_rows = [r for rres in cres["regimes"].values() for r in rres["measurements"]]
        seeds = sorted({int(r["seed"]) for r in all_rows if "baseline" in r.get("arms", {})})
        ha, hb = seeds[: len(seeds) // 2], seeds[len(seeds) // 2 :]
        in_a = [r for r in all_rows if int(r["seed"]) in ha]
        in_b = [r for r in all_rows if int(r["seed"]) in hb]
        pooled_pit = RE.cluster_interval(
            _abs_delta(per_window(in_a, "baseline", "pit"), per_window(in_b, "baseline", "pit")),
            alpha=alpha,
            seed=seed,
        )
        pooled_ltas = RE.cluster_interval(
            _abs_delta(per_window(in_a, "baseline", "ltas"), per_window(in_b, "baseline", "ltas")),
            alpha=alpha,
            seed=seed,
        )
        per_regime["_cohort"] = dict(
            gate=cres["plan"].get("gate"),
            recordings=cres["plan"].get("recordings"),
            block_null_variation=dict(
                seed_halves=[ha, hb],
                n_blocks=len(set(per_window(in_a, "baseline", "pit"))),
                pit_tolerance=pooled_pit.upper,
                pit_interval=pooled_pit.as_dict(),
                ltas_tolerance_db=pooled_ltas.upper,
                ltas_interval=pooled_ltas.as_dict(),
                composite_tolerance=float(man["gates"]["composite_tolerance"]),
                conditioning=(
                    "pooled over every held-out block of this cohort's recording(s), across "
                    "regimes — conditional on that recording, never a population claim"
                ),
            ),
        )
        out["cohorts"][name] = per_regime
    return out


def candidate_leakage_guard(
    cand_spec: dict[str, Any],
    scored: list[RE.Window],
    *,
    cohort_name: str,
    rig: str | None = None,
) -> dict[str, Any]:
    """Refuse a candidate whose fit supports touch a scored support.

    Runs BEFORE the arm is measured. For a revised export the provenance is
    read from the file itself (``training_provenance.clips`` plus its
    ``manifest_sha256``, which must equal the frozen fit manifest the
    evaluation manifest names). A legacy candidate must declare its training
    windows in the manifest; an undeclared one is refused rather than assumed
    clean.
    """
    kind = str(cand_spec.get("kind"))
    guard = float(cand_spec.get("training_guard_seconds", RE.OBS_N_FFT / RE.SR))
    if kind == "revised_export":
        export_path, expected_sha = _resolve_candidate_export(
            cand_spec, rig, context=f"cohort {cohort_name} candidate"
        )
        cand = RE.read_candidate_export(export_path)
        if rig is not None and cand.rig_id != rig:
            die(
                f"cohort {cohort_name}: candidate export rig_id is {cand.rig_id!r}, "
                f"cohort rig is {rig!r}"
            )
        training = cand.training_windows()
        prov = cand.provenance()
        expected = expected_sha
        if expected is None:
            die(
                f"cohort {cohort_name}: the candidate arm must name the frozen fit manifest as "
                '"fit_manifest_sha256" so the export can be pinned to it'
            )
        if str(prov.get("fit_manifest_sha256")) != str(expected):
            die(
                f"candidate export was fitted under manifest "
                f"{prov.get('fit_manifest_sha256')!r}, the evaluation manifest requires "
                f"{expected!r}"
            )
        refined = [k for k in prov["training_rps_keys"] if k == RE.REFINED_RPS_KEY]
        report = RE.training_leakage(training, scored, guard_seconds=guard)
        report["candidate"] = prov
        report["training_used_refined_labels"] = refined
    else:
        declared = [
            RE.Window(
                str(w["recording"]),
                float(w["start_s"]),
                float(w["seconds"] if "seconds" in w else w["duration_s"]),
                regime=str(w.get("regime", "unknown")),
                role="candidate_training",
            )
            for w in (cand_spec.get("training_provenance") or {}).get("clips", [])
        ]
        if not declared and not bool(cand_spec.get("no_training_supports")):
            die(
                f"cohort {cohort_name}: the candidate arm declares no training supports. Declare "
                '"training_provenance": {"clips": [...]} or, for an arm that was never fitted '
                '(a control), "no_training_supports": true'
            )
        report = RE.training_leakage(declared, scored, guard_seconds=guard)
        report["candidate"] = dict(
            kind=kind,
            label=cand_spec.get("label"),
            declared_training_clips=[w.as_dict() for w in declared],
            no_training_supports=bool(cand_spec.get("no_training_supports")),
        )
    if not report["clean"]:
        die(
            f"cohort {cohort_name}: the candidate's training supports intersect scored held-out "
            f"supports: {report['overlaps']}"
        )
    return report


def run_check(
    man: dict[str, Any], out: Path, *, n_mics: int | None, provenance: dict[str, Any] | None = None
) -> int:
    cal_path = out / "calibration.json"
    if not cal_path.is_file():
        die(f"{cal_path} not found — run --prepare first; --check never invents a calibration")
    cal = json.loads(cal_path.read_text())
    arms = dict(man["arms"])
    if "candidate" not in arms:
        die("manifest has no 'candidate' arm: nothing to check")
    cand_spec = dict(arms["candidate"])
    tracker = Tracker(dict(man["scorer"]))
    now = frozen_inputs(man, tracker.record)
    verify_frozen_inputs(now, dict(cal.get("frozen_inputs") or {}))
    protocol_sha = verify_protocol_fingerprint(man, cal)
    seeds = [int(s) for s in require(dict(man["null_variation"]), "seeds", "null_variation")]

    alpha = float(man["gates"]["alpha"])
    bseed = int(man["gates"]["bootstrap_seed"])
    report: dict[str, Any] = dict(
        mode="check",
        schema=RE.SCHEMA,
        manifest=dict(
            path=man["_path"], sha256=man["_digest"], protocol_sha256=protocol_sha
        ),
        import_provenance=provenance,
        scorer=tracker.record,
        frozen_inputs=now,
        gates_config=man["gates"],
        candidate_arm=cand_spec,
        render_seeds=seeds,
        caveat=RE.ADAPTIVE_SELECTION_CAVEAT,
        cohorts={},
    )
    gates: list[RE.Gate] = []
    for cohort in man["cohorts"]:
        name = str(cohort["name"])
        rig = str(cohort.get("rig") or name)
        cal_c = dict(cal["calibration"]["cohorts"].get(name) or {})
        if not cal_c:
            die(f"calibration has no cohort {name!r}")
        bundles_by_regime = all_families(cohort)
        required = [str(r) for r in (cohort.get("recordings") or [])]
        required_regimes = [str(rs["regime"]) for rs in cohort["regimes"]]
        cres: dict[str, Any] = dict(regimes={})
        blocks: dict[str, list[dict[str, Any]]] = {}
        all_reports: list[dict[str, Any]] = []
        measured: list[str] = []
        cand_pit_blocks: dict[str, float] = {}
        base_pit_blocks: dict[str, float] = {}
        cand_ltas_blocks: dict[str, float] = {}
        base_ltas_blocks: dict[str, float] = {}
        cand_comp_blocks: dict[str, float] = {}
        base_comp_blocks: dict[str, float] = {}
        block_regime: dict[str, str] = {}
        for regime_spec in cohort["regimes"]:
            regime = str(regime_spec["regime"])
            cal_r = dict(cal_c.get(regime) or {})
            if not cal_r:
                die(f"calibration has no regime {regime!r} for cohort {name!r}")
            calibration = [
                w for b in bundles_by_regime[regime].values() for w in b.calibration_windows()
            ]
            windows, reports = resolve_windows(cohort, regime_spec, calibration)
            all_reports.extend(reports)
            leakage = candidate_leakage_guard(cand_spec, windows, cohort_name=name, rig=rig)

            def model_for(
                arm: str, spec: dict[str, Any], w: RE.Window, _r: str = regime
            ) -> ArmModel:
                return arm_model(arm, spec, regime=_r, recording=w.recording, rig=rig)

            rows = measure_arm_set(
                cohort,
                regime_spec,
                windows,
                tracker=tracker,
                arms={"candidate": cand_spec},
                model_for=model_for,
                seeds=seeds,
                n_mics=n_mics,
                tag=name,
            )
            cand_pit = per_recording(rows, "candidate", "pit")
            cand_ltas = per_recording(rows, "candidate", "ltas")
            cand_comp = RE.composite_score(frame_scores(rows, "candidate"))
            measured.extend(cand_pit)
            for row in rows:
                block_regime[row["window"]["key"]] = regime
            cand_pit_blocks |= per_window(rows, "candidate", "pit")
            cand_ltas_blocks |= per_window(rows, "candidate", "ltas")
            cand_comp_blocks |= composite_per_window(rows, "candidate")
            base_pit_blocks |= dict(cal_r.get("baseline_pit_by_window") or {})
            base_ltas_blocks |= dict(cal_r.get("baseline_ltas_abs_db_by_window") or {})
            base_comp_blocks |= dict(cal_r.get("baseline_composite_by_window") or {})
            cres["regimes"][regime] = dict(
                frozen_family=cal_r["frozen_family"],
                support_reports=reports,
                leakage_guard=leakage,
                measurements=rows,
                candidate_pit=cand_pit,
                candidate_pit_by_window=per_window(rows, "candidate", "pit"),
                candidate_ltas_abs_db=cand_ltas,
                candidate_composite=cand_comp,
            )
            if str(cohort.get("gate") or "") == "dregon_pit" and regime == "cruise":
                real, base = dict(cal_r["real_pit"]), dict(cal_r["baseline_pit"])
                paired = sorted(set(real) & set(base) & set(cand_pit))
                gates.append(
                    RE.dregon_pit_gate(
                        {
                            r: dict(real=real[r], baseline=base[r], candidate=cand_pit[r])
                            for r in paired
                        },
                        gap_fraction=float(man["gates"]["dregon_gap_fraction"]),
                        alpha=alpha,
                        required_recordings=required,
                        seed=bseed,
                    )
                )
                gates.append(
                    RE.composite_delta_gate(
                        dict(cal_r["baseline_composite_per_recording"]),
                        {r: float(v["score"]) for r, v in cand_comp["per_recording"].items()},
                        alpha=alpha,
                        tolerance=float(man["gates"]["composite_tolerance"]),
                        seed=bseed,
                        name="dregon_held_out_composite_delta",
                    )
                )
                gates.append(
                    RE.ltas_gate(
                        dict(cal_r["baseline_ltas_abs_db"]),
                        cand_ltas,
                        tolerance_db=cal_r["null_variation"]["recording_level"][
                            "ltas_tolerance_db"
                        ],
                        name="dregon_baseline_calibrated_ltas",
                    )
                )
            if str(cohort.get("gate") or "") == "michaels_ratio":
                base_w = dict(cal_r["baseline_pit_by_window"])
                cand_w = per_window(rows, "candidate", "pit")
                shared = sorted(set(base_w) & set(cand_w))
                blocks[regime] = [
                    dict(key=k, baseline=base_w[k], candidate=cand_w[k]) for k in shared
                ]
                cres["regimes"][regime]["unpaired_blocks"] = dict(
                    calibration_only=sorted(set(base_w) - set(cand_w)),
                    candidate_only=sorted(set(cand_w) - set(base_w)),
                )
        gates.append(
            RE.cohort_completeness_gate(
                required=required,
                measured=measured,
                support_reports=all_reports,
                name=f"{name}_cohort_complete",
            )
        )
        if str(cohort.get("gate") or "") == "michaels_ratio":
            recording = str((cohort.get("recordings") or ["FLY124"])[0])
            gates.append(
                RE.michaels_ratio_gate(
                    blocks,
                    ratio_max=float(man["gates"]["michaels_ratio_max"]),
                    alpha=alpha,
                    seed=bseed,
                    recording=recording,
                    required_regimes=required_regimes,
                )
            )
            pooled = dict(dict(cal_c.get("_cohort") or {}).get("block_null_variation") or {})
            ltas_tol = pooled.get("ltas_tolerance_db")
            gates.append(
                RE.conditional_non_regression_gate(
                    base_comp_blocks,
                    cand_comp_blocks,
                    tolerance=float(man["gates"]["composite_tolerance"]),
                    alpha=alpha,
                    seed=bseed,
                    name="michaels_fly124_composite_non_regression",
                    quantity="composite spectral risk (nats/s of unique support)",
                    recording=recording,
                    regimes=block_regime,
                    required_regimes=required_regimes,
                )
            )
            gates.append(
                RE.conditional_non_regression_gate(
                    base_ltas_blocks,
                    cand_ltas_blocks,
                    tolerance=ltas_tol,
                    alpha=alpha,
                    seed=bseed,
                    name="michaels_fly124_absolute_ltas_non_regression",
                    quantity="absolute-level LTAS band error (dB)",
                    recording=recording,
                    regimes=block_regime,
                    required_regimes=required_regimes,
                )
            )
            cres["block_pairing"] = dict(
                baseline_pit=base_pit_blocks,
                candidate_pit=cand_pit_blocks,
                baseline_ltas_abs_db=base_ltas_blocks,
                candidate_ltas_abs_db=cand_ltas_blocks,
                baseline_composite=base_comp_blocks,
                candidate_composite=cand_comp_blocks,
                ltas_tolerance_db=ltas_tol,
            )
        report["cohorts"][name] = cres

    decisions = dict(
        mode="check",
        manifest=dict(
            path=man["_path"], sha256=man["_digest"], protocol_sha256=protocol_sha
        ),
        scorer=tracker.record,
        frozen_inputs=now,
        candidate_arm=cand_spec,
        render_seeds=seeds,
        caveat=RE.ADAPTIVE_SELECTION_CAVEAT,
        all_passed=bool(gates) and all(g.passed for g in gates),
        gates=[g.as_dict() for g in gates],
    )
    RE.write_json(out / "metrics.json", report)
    RE.write_json(out / "gates.json", decisions)
    print(json.dumps(RE.json_ready(decisions), indent=1))
    print(f"wrote {out / 'metrics.json'} and {out / 'gates.json'}")
    if not decisions["all_passed"]:
        print(f"GATES UNMET: {[g.name for g in gates if not g.passed] or ['no gate evaluated']}")
        return 1
    return 0


def run_verify_adapter(
    man: dict[str, Any],
    out: Path,
    *,
    n_mics: int | None,
    seeds: int,
    provenance: dict[str, Any] | None = None,
) -> int:
    """Check each baseline predicted-M adapter against its renderer's mean statistics.

    Required by the frozen protocol: an old-model adapter must predict the same
    observation its own renderer produces. Both are now in PHYSICAL units — the
    render runs with ``normalize_rms=None`` and the declared rate-factor
    conversion — so the level offset is meaningful rather than arbitrary, and
    two disjoint seed halves of the SAME parameters give the Monte-Carlo floor
    the agreement must be read against.
    """
    rows: list[dict[str, Any]] = []
    for cohort in man["cohorts"]:
        bundles_by_regime = all_families(cohort)
        for regime_spec in cohort["regimes"]:
            regime = str(regime_spec["regime"])
            bundles = bundles_by_regime[regime]
            calibration = [w for b in bundles.values() for w in b.calibration_windows()]
            windows, _ = resolve_windows(cohort, regime_spec, calibration)
            for family in bundles:
                for w in windows[:1]:
                    mp = baseline_params(
                        cohort,
                        regime,
                        w.recording,
                        family=family,
                        bundles_by_regime=bundles_by_regime,
                    )
                    physical = RE.to_renderer_units(mp)
                    clip = RE.load_window(
                        w,
                        dataset=str(cohort["dataset"]),
                        version=cohort.get("version"),
                        channels=cohort.get("channels"),
                        rps_key=RE.assert_raw_reference(str(cohort["scoring_rps_key"])),
                    )
                    mics = int(n_mics or clip.audio.shape[0])
                    reference = np.asarray(clip.rps, dtype=np.float64)
                    pg = RE.window_periodogram(clip)
                    band = RE.observation_band(pg.freqs)
                    pred = RE.predicted_m(mp, pg, n_mics=mics)
                    halves: list[np.ndarray] = []
                    for grp in range(2):
                        acc = None
                        for j in range(seeds):
                            audio = S2.render_from_export(
                                dict(physical.params),
                                reference,
                                sample_rate_work=RE.SAMPLE_RATE_WORK,
                                n_mics=mics,
                                seed=7000 + 100 * grp + j,
                                normalize_rms=None,
                            )
                            cl = Clip(
                                "syn",
                                "synthetic",
                                np.asarray(audio, dtype=np.float32),
                                reference[:, : np.asarray(audio).shape[-1]],
                                RE.SR,
                            )
                            p = np.asarray(RE.window_periodogram(cl).power, dtype=np.float64)
                            acc = p if acc is None else acc + p
                        assert acc is not None
                        halves.append(acc / seeds)
                    n = min(halves[0].shape[1], pred.shape[1])

                    def avg(x: np.ndarray, _n: int = n) -> np.ndarray:
                        return x[:, :_n].mean(axis=1)[:, None, :]

                    rows.append(
                        dict(
                            cohort=cohort["name"],
                            regime=regime,
                            family=family,
                            window=w.as_dict(),
                            mics=mics,
                            seeds_per_half=seeds,
                            render_units=physical.source["render_units"],
                            observation_law=RE.HISTORICAL_FORWARD_LABEL,
                            adapter_vs_renderer=RE.adapter_agreement(
                                avg(pred), avg((halves[0] + halves[1]) / 2), band
                            ),
                            monte_carlo_floor=RE.adapter_agreement(
                                avg(halves[0]), avg(halves[1]), band
                            ),
                        )
                    )
                    print(json.dumps(RE.json_ready(rows[-1]), indent=1), flush=True)
    payload = dict(
        mode="verify-adapter",
        manifest=dict(path=man["_path"], sha256=man["_digest"]),
        import_provenance=provenance,
        note=(
            "both sides are in physical units (normalize_rms=None plus the declared rate "
            "factor), so the level offset is a real disagreement; the Monte-Carlo floor is two "
            "disjoint seed halves of the SAME parameters"
        ),
        rows=rows,
    )
    RE.write_json(out / "adapter_verification.json", payload)
    print(f"wrote {out / 'adapter_verification.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="revised-phase baseline/candidate gate runner")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--plan", action="store_true", help="resolve cohorts and supports, no scoring")
    ap.add_argument("--prepare", action="store_true", help="measure the baseline arm and calibrate")
    ap.add_argument("--check", action="store_true", help="measure the candidate arm and gate it")
    ap.add_argument("--verify-adapter", action="store_true", help="adapter vs renderer statistics")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: manifest out_dir)")
    ap.add_argument("--mics", type=int, default=None, help="limit the microphones (smoke runs)")
    ap.add_argument("--adapter-seeds", type=int, default=3, help="render seeds per half")
    args = ap.parse_args(argv)

    if sum(bool(m) for m in (args.plan, args.prepare, args.check, args.verify_adapter)) != 1:
        die("choose exactly one of --plan / --prepare / --check / --verify-adapter")
    provenance = RE.import_provenance(expected_root=Path.cwd())
    man = load_manifest(args.manifest)
    out = Path(args.out or require(man, "out_dir", str(args.manifest)))
    out.mkdir(parents=True, exist_ok=True)
    if args.plan:
        return run_plan(man, out, provenance=provenance)
    if args.prepare:
        return run_prepare(man, out, n_mics=args.mics, provenance=provenance)
    if args.check:
        return run_check(man, out, n_mics=args.mics, provenance=provenance)
    return run_verify_adapter(
        man, out, n_mics=args.mics, seeds=int(args.adapter_seeds), provenance=provenance
    )


if __name__ == "__main__":
    sys.exit(main())
