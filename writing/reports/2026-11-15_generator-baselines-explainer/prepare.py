"""Build explainer assets from saved results only; never fit or evaluate a model.

Both panels and all current arms must be present in the neural extension evidence.
Archived methods are excluded from tables, controls and listening.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from plots.generator_paper import (  # noqa: E402
    ARCHIVED_METHODS,
    MATCHED_ONLY_METHODS,
    REGIME_SUFFIX,
    listening_spectrograms,
    tone_variation_spectrum,
)

OURS = {
    "stochastic": "Our full model",
    "stochastic_coherent": "Our model: no added tone variation",
}
# The original arms: ours plus each external family fitted once on the pooled windows.
POOLED = {
    **OURS,
    "yang2025": "Pulses: integer clock",
    "yang2025_phase": "Pulses: fractional carry",
    "lee2026_gp": "GP: white noise",
    "lee2026_gp_colored": "GP: coloured noise",
    "doerfler2026_eone": "EONE-RPS curve bank",
}
# One expert per regime on our core supports, power-blended.
MATCHED = {
    "yang2025_regimes": "Pulses: integer clock, two experts",
    "yang2025_phase_regimes": "Pulses: fractional carry, two experts",
    "lee2026_gp_regimes": "GP: white noise, two experts",
    "lee2026_gp_colored_regimes": "GP: coloured noise, two experts",
    "doerfler2026_eone_regimes": "EONE-RPS curve bank, two experts",
    "doerfler2026_ptr_regimes": "PTR, two experts",
    "doerfler2026_hpn_regimes": "HPN, two experts",
    "renault2023_piano_regimes": "DDSP-Piano, two experts",
}
METHODS = {**POOLED, **MATCHED}
PAIRS = {m: m.removesuffix(REGIME_SUFFIX) for m in MATCHED if m not in MATCHED_ONLY_METHODS}
SYNTH_ANCHORS = {
    "stochastic": "stochastic",
    "stochastic_coherent": "stochastic",
    "yang2025": "yang",
    "yang2025_phase": "yang",
    "lee2026_gp": "gp",
    "lee2026_gp_colored": "gp",
    "doerfler2026_eone": "eone",
    "doerfler2026_ptr": "engine",
    "doerfler2026_hpn": "engine",
    "renault2023_piano": "piano",
}
REGIMES = ("cruise", "standby", "transition")
SCORE_REGIMES = (*REGIMES, "all")
REGIME_NAMES = {
    "cruise": "Fast rotors",
    "standby": "Slow rotors",
    "transition": "Changing speeds",
    "all": "All windows",
}
EXPERTS = ("standby", "cruise")
# Blend thresholds shared with our renderer (render.regime_blend_weight).
BLEND_RPS = {"standby_max_rps": 45.0, "cruise_min_rps": 65.0}
# Listening presentation only (codec fidelity, not a model or metric criterion).
PCM16_TOLERANCE = 0.01
PCM16_RULE = (
    "reference_relative_rms_error = ||decode(encode(g*ref)) - g*ref|| / ||g*ref||, with PCM16 coding "
    "rint(x*32767) and decoding /32767, at g = 0.8/max(reference peak, candidate peak); a candidate above "
    "the tolerance cannot share a gain with the recording and becomes a standalone preview"
)
SUPPORT = re.compile(r"(FLY\d+)@(\d+(?:\.\d+)?)\+(\d+(?:\.\d+)?)")
# Fixed EONE design values that the guide states in prose and equations.
EONE_PROSE_SETTINGS = {
    "sample_rate": 16000,
    "piece_s": 4.0,
    "n_orders": 128,
    "n_bands": 64,
    "n_knots": 40,
    "rpm_max": 10000.0,
    "hop": 256,
    "noise_loop": 65536,
    "alias_transition_hz": 100.0,
    "epochs": 50,
    "max_lr": 0.01,
    "weight_decay": 0.0,
    "mrstft_max_fft": 32768,
    "mrstft_min_fft": 32,
    "log_floor_rel": 1e-3,
    "harmonic_n_fft": 65536,
    "harmonic_win": 16384,
    "harmonic_hop": 256,
    "harmonic_halfwidth_bins": 4,
    "energy_floor_rel": 1e-6,
    "harmonic_weight": 1.0,
    "curvature_weight": 1e-3,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_panels(
    evidence: dict,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], dict[str, str]]:
    """Require all current sources and follow the evidence's order and panel membership."""
    assert evidence["schema"] == "generator-paper-evidence/1"
    sources = evidence["method_sources"]
    declared_order = evidence["method_order"]
    assert sorted(declared_order) == sorted(sources), (
        "method_order does not list exactly the method sources"
    )
    arms = tuple(m for m in declared_order if m not in ARCHIVED_METHODS)
    assert set(arms) == set(METHODS), f"unexpected current arms: {sorted(set(arms) ^ set(METHODS))}"
    declared = evidence["regime_comparison"]

    def current(key: str) -> tuple[str, ...]:
        return tuple(m for m in declared[key] if m not in ARCHIVED_METHODS)

    native, primary, matched = (
        current(k) for k in ("pooled_panel_methods", "primary_methods", "matched_methods")
    )
    assert sorted(native) == sorted(POOLED) and sorted(matched) == sorted(MATCHED)
    assert primary == (*OURS, *matched)
    pairs = {
        m: base
        for m, base in declared["method_pairs"].items()
        if m not in ARCHIVED_METHODS and base not in ARCHIVED_METHODS
    }
    assert pairs == PAIRS, "method_pairs must include the paired variants only"
    assert all(
        sources[m]["fitting_treatment"] == "standby_cruise_matched"
        and sources[m].get("pooled_method") == pairs.get(m)
        for m in matched
    )
    return arms, native, primary, pairs


def support_window(key: str) -> tuple[str, float, float]:
    """(recording, start s, length s) of a support key such as ``md2:FLY125@2.000+4``."""
    match = SUPPORT.search(key)
    assert match, f"no support window in {key!r}"
    recording, start, length = match.groups()
    return recording, float(start), float(length)


def core_supports() -> tuple[dict[str, dict], str]:
    """Exact standby/cruise supports of our accepted fits, read from the hash-checked inputs."""
    provenance_path = ROOT / "results/noise_generator_paper/input_provenance.json"
    supports = {}
    for fit in json.loads(provenance_path.read_text())["accepted_md2_fits"]:
        path = ROOT / fit["local_path"]
        assert digest(path) == fit["sha256"], path
        members = json.loads(path.read_text())["pool"]["members"]
        supports[fit["regime"]] = {
            "accepted_fit": fit["local_path"],
            "sha256": fit["sha256"],
            "members": members,
            "windows": sorted(support_window(m) for m in members),
        }
    assert supports.keys() == set(EXPERTS)
    return supports, digest(provenance_path)


def support_table(supports: dict, pooled_windows: list[dict]) -> str:
    lines = [
        "| Fit | Data in FLY125 (s, recording clock) | Total audio |",
        "|:--|:--|--:|",
    ]
    for regime, title in (
        ("standby", "Standby expert (slow rotors)"),
        ("cruise", "Cruise expert (fast rotors)"),
    ):
        windows = supports[regime]["windows"]
        assert {r for r, _, _ in windows} == {"FLY125"}
        spans = ", ".join(f"{s:g}–{s + d:g}" for _, s, d in windows)
        total = sum(d for _, _, d in windows)
        lines.append(f"| {title} | {len(windows)} windows: {spans} | {total:g} s |")
    lengths = {
        w["provenance"]["native_n_samples"] / w["provenance"]["native_sr"] for w in pooled_windows
    }
    assert len(lengths) == 1
    length = lengths.pop()
    lines.append(
        f"| Native pooled baseline fit | {len(pooled_windows)} consecutive {length:g}-s windows, "
        f"all speed regimes; GP selects a subset | {len(pooled_windows) * length:g} s |"
    )
    return "\n".join(lines) + "\n"


def check_matched(method: str, raw: dict, supports: dict) -> dict:
    """Require the exact core supports, the shared blend rule and frozen expert artifacts."""
    report = raw["calibration"]
    assert report["fitting_treatment"] == "standby_cruise_matched"
    assert report["method"] == method and report["base_method"] == method.removesuffix(
        REGIME_SUFFIX
    )
    windows = report["calibration_windows"]
    assert [w["key"] for w in raw["calibration_windows"]] == [w["key"] for w in windows]
    assert len(windows) == sum(len(supports[r]["windows"]) for r in EXPERTS)
    for regime in EXPERTS:
        got = sorted(support_window(w["key"]) for w in windows if w["fit_regime"] == regime)
        assert got == supports[regime]["windows"], (method, regime, got)
    blend = report["blend"]
    assert {k: float(blend[k]) for k in BLEND_RPS} == BLEND_RPS, blend
    assert report["regimes"].keys() == set(EXPERTS)
    for r, expert in report["regimes"].items():
        assert sorted(w["key"] for w in expert["fit_inputs"]) == sorted(
            w["key"] for w in windows if w["fit_regime"] == r
        )
        assert expert["artifacts"] and all(
            len(a["sha256"]) == 64 for a in expert["artifacts"].values()
        )
    return {
        "manifest_sha256": report["manifest_sha256"],
        "blend": blend,
        "calibration_windows": [
            {k: w[k] for k in ("key", "regime", "fit_regime")} for w in windows
        ],
        "experts": {
            r: {
                "artifacts": e["artifacts"],
                "fit_seed": e["fit_seed"],
                "fit_device": e["fit_device"],
                "fit_wall_seconds": e["fit_wall_seconds"],
                "fit_inputs": [
                    {k: i[k] for k in ("key", "regime", "fit_regime")} for i in e["fit_inputs"]
                ],
            }
            for r, e in report["regimes"].items()
        },
    }


def expert_summary(method: str, expert: dict) -> str:
    """One table cell from a variant's own fit report for one expert."""
    fit = expert["fit"]
    if method in MATCHED_ONLY_METHODS:
        assert fit["production_settings"] and fit["status"] == "completed", (
            f"{method}: not a completed production fit"
        )
        if method == "renault2023_piano_regimes":
            assert fit["total_updates"] == 6000
            budget = f"{fit['total_updates']} updates in three phases"
            commit = fit["upstream_commit"]
        else:
            assert fit["epochs"] == 100
            budget = (
                f"{fit['epochs']} epochs; {fit['optimizer_steps_per_mic']} updates per microphone"
            )
            commit = fit["upstream"]["commit"]
        assert len(commit) == 40, f"{method}: no pinned upstream revision"
        return f"{budget}; upstream `{commit[:12]}…`"
    base = method.removesuffix(REGIME_SUFFIX)
    if base.startswith("yang2025"):
        return (
            f"Selected Δμ = {fit['modulation_range_hz']:g} Hz, ratio {fit['white_ratio']:g}; "
            f"{fit['calibration_segments']} × 4-s fit segments"
        )
    if base.startswith("lee2026_gp"):
        return (
            f"{fit['frame_manifest']['selected_frames']} × {fit['settings']['frame_s']:g}-s frames; "
            f"checkpoint `{fit['checkpoint_sha256'][:12]}…`"
        )
    assert base == "doerfler2026_eone"
    training = fit["training"]
    assert training["completed_epochs"] == training["planned_epochs"] == fit["settings"]["epochs"]
    return (
        f"{fit['pieces']['n']} × {fit['pieces']['piece_s']:g}-s pieces; selected epoch "
        f"{training['best_epoch']} of {fit['settings']['epochs']}"
    )


def matched_fit_table(raws: dict, matched: tuple[str, ...]) -> str:
    gp = [
        raws[m]["calibration"]["regimes"]
        for m in ("lee2026_gp_regimes", "lee2026_gp_colored_regimes")
    ]
    for regime in EXPERTS:
        assert (
            gp[0][regime]["fit"]["checkpoint_sha256"] == gp[1][regime]["fit"]["checkpoint_sha256"]
        )
    lines = [
        "| Two-expert arm | Standby expert | Cruise expert | Frozen manifest |",
        "|:--|:--|:--|:--|",
    ]
    for method in matched:
        report = raws[method]["calibration"]
        cells = [expert_summary(method, report["regimes"][r]) for r in EXPERTS]
        lines.append(
            f"| {METHODS[method]} | {cells[0]} | {cells[1]} | `{report['manifest_sha256'][:12]}…` |"
        )
    return "\n".join(lines) + "\n"


def worker_path_differences(
    a: object, b: object, where: str = "provenance"
) -> list[dict[str, str]]:
    """Require equal window provenance, except absolute worker filesystem paths.

    Raws from different hosts (Kaggle, university cluster) may record the same
    pinned file under different absolute paths. Every other key, value and
    list entry must agree exactly; path differences are returned for the record.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), f"{where}: keys differ: {sorted(a.keys() ^ b.keys())}"
        return [d for k in a for d in worker_path_differences(a[k], b[k], f"{where}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"{where}: list lengths differ"
        return [
            d
            for i, (x, y) in enumerate(zip(a, b))
            for d in worker_path_differences(x, y, f"{where}[{i}]")
        ]
    if a == b:
        return []
    assert (
        isinstance(a, str)
        and isinstance(b, str)
        and a.startswith("/")
        and b.startswith("/")
        and Path(a).name == Path(b).name
    ), f"{where}: semantic provenance mismatch {a!r} != {b!r}"
    return [{"field": where, "reference_owner": a, "method": b}]


def listening_controls(arms: tuple[str, ...]) -> list[str]:
    """Case and method selectors; groups follow the fitting treatment of the arms present."""
    html = [
        '<div class="listening-controls" hidden>',
        '<label for="listening-case">Speed case',
        '<select id="listening-case">',
        *(
            f'<option value="{r}"{" selected" if r == "cruise" else ""}>{REGIME_NAMES[r]} ({r})</option>'
            for r in REGIMES
        ),
        "</select>",
        "</label>",
        '<label for="listening-method">Generated method',
        '<select id="listening-method">',
    ]
    groups = [("Our model", tuple(OURS))]
    groups.append(("Baselines: two experts, our supports", tuple(m for m in arms if m in MATCHED)))
    groups.append(
        ("Baselines: native pooled fit", tuple(m for m in arms if m in POOLED and m not in OURS))
    )
    for title, methods in groups:
        html.append(f'<optgroup label="{escape(title)}">')
        html.extend(
            f'<option value="{m}"{" selected" if m == "stochastic" else ""}>{escape(METHODS[m])}</option>'
            for m in methods
        )
        html.append("</optgroup>")
    html.extend(["</select>", "</label>", "</div>"])
    return html


def pcm16_relative_error(reference: np.ndarray, gain: float) -> float:
    """Relative RMS error of the recording after scaling by ``gain`` and PCM16 coding."""
    scaled = reference.astype(np.float64) * gain
    decoded = np.rint(scaled * 32767).astype(np.int16).astype(np.float64) / 32767
    return float(np.linalg.norm(decoded - scaled) / np.linalg.norm(scaled))


def level_domain(scored: dict) -> dict:
    """Whether a scored draw-0 output stays inside the metric's level domain.

    A scientific scoring property, separate from playback: reads the evaluator's
    own record for this pair (``details.nominal80_psychoacoustics``); at the fixed
    nominal gain, a generated 25–250 Hz third-octave level above the stated limit
    on any microphone leaves the psychoacoustic descriptors undefined. Raws from
    the earlier metric version carry no low-band record; that version's loudness
    kernel refused out-of-domain input, so a defined loudness on every microphone
    means the output was inside the domain.
    """
    assert scored["details"] is not None, (
        "listening windows must be the evaluator's stored example windows"
    )
    record = scored["details"]["nominal80_psychoacoustics"]
    if "zwicker_low_band_max_gen_db_per_mic" not in record:
        loudness = record["zwicker_loudness_gen_sone_per_mic"]
        assert all(v is not None and np.isfinite(v) for v in loudness), record
        return {
            "domain_valid": True,
            "reason": None,
            "mics_out_of_domain": [],
            "logged_undefined_reasons": [],
            "source": "legacy metric record: loudness defined on every microphone",
        }
    limit = float(record["zwicker_low_band_limit_db"])
    generated = [float(v) for v in record["zwicker_low_band_max_gen_db_per_mic"]]
    valid = [bool(v) for v in record["zwicker_domain_valid_per_mic"]]
    out = [m for m, level in enumerate(generated) if level > limit]
    assert not any(valid[m] for m in out), "domain flags disagree with the stored low-band levels"
    # The evaluator logs why the psychoacoustic descriptors are undefined; a
    # generated domain failure must appear there with the ISO 532-1 reason.
    logged = sorted({r for k, r in scored["undefined"].items() if k.startswith("nominal80__")})
    if out:
        assert logged and all("ISO 532-1:2017 A.3" in r for r in logged), logged
    reason = (
        f"generated 25-250 Hz third-octave level above {limit:g} dB at the fixed nominal scoring gain "
        f"on microphones {out}; scored metric descriptors undefined"
        if out
        else None
    )
    return {
        "domain_valid": not out,
        "reason": reason,
        "limit_db": limit,
        "low_band_max_gen_db_per_mic": generated,
        "low_band_max_ref_db_per_mic": [
            float(v) for v in record["zwicker_low_band_max_ref_db_per_mic"]
        ],
        "domain_valid_per_mic": valid,
        "mics_out_of_domain": out,
        "logged_undefined_reasons": logged,
        "source": "raw pair draw 0: details.nominal80_psychoacoustics and undefined",
    }


def prepare_listening(assets: Path, evidence: dict, raws: dict, arms: tuple[str, ...]) -> dict:
    """Export the predeclared draw-0 cases, never regenerate a model output."""
    out = assets / "listening"
    out.mkdir(exist_ok=True)
    labels = {"reference": "Real recording", **{m: METHODS[m] for m in arms}}
    # Remove only obsolete cards created by earlier explainer builds, never any raw input.
    for method in ARCHIVED_METHODS:
        for regime in REGIMES:
            for ext in ("wav", "png"):
                (out / f"{regime}__{method}.{ext}").unlink(missing_ok=True)
    owner = raws["stochastic"]
    html, manifest = listening_controls(arms), {}
    for regime in REGIMES:
        title = REGIME_NAMES[regime]
        pair = next(p for p in owner["pairs"] if p["regime"] == regime and p["draw"] == 0)
        key = pair["key"]
        index = next(i for i, w in enumerate(owner["evaluation_windows"]) if w["key"] == key)
        window = owner["evaluation_windows"][index]
        reference_path = ROOT / evidence["reference_examples_root"] / f"reference__{index:03d}.npz"
        with np.load(reference_path) as reference:
            assert str(reference["key"]) == key and str(reference["regime"]) == regime
            sr = int(reference["sr"])
            rps = reference["rps"]
            real = reference["real"]
            waves = {"reference": real[0]}
        sources = {
            "reference": {
                "path": str(reference_path.relative_to(ROOT)),
                "sha256": digest(reference_path),
            }
        }
        path_differences, domains, verified_refs = {}, {}, {}
        for method in arms:
            method_index = next(
                i for i, w in enumerate(raws[method]["evaluation_windows"]) if w["key"] == key
            )
            path_differences[method] = worker_path_differences(
                window["provenance"], raws[method]["evaluation_windows"][method_index]["provenance"]
            )
            scored = next(p for p in raws[method]["pairs"] if p["key"] == key and p["draw"] == 0)
            assert scored["regime"] == regime and scored["start_s"] == pair["start_s"]
            path = (
                ROOT
                / evidence["method_sources"][method]["examples_root"]
                / f"{method}__{method_index:03d}.npz"
            )
            if method in MATCHED_ONLY_METHODS:
                own_ref = path.with_name(f"reference__{method_index:03d}.npz")
                with np.load(own_ref) as own:
                    assert (
                        str(own["key"]) == key
                        and str(own["regime"]) == regime
                        and int(own["sr"]) == sr
                    ), own_ref
                    assert np.array_equal(own["rps"], rps) and np.array_equal(own["real"], real), (
                        own_ref
                    )
                verified_refs[method] = {
                    "path": str(own_ref.relative_to(ROOT)),
                    "sha256": digest(own_ref),
                }
            with np.load(path) as generated:
                assert str(generated["key"]) == key and int(generated["sr"]) == sr
                samples = generated["generated"]
                assert samples.shape == real.shape == (8, 4 * sr)
                waves[method] = samples[0].copy()
            sources[method] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
            domains[method] = level_domain(scored)
        assert sr == 16000 and rps.shape == (4, 4 * sr)
        assert all(w.shape == (4 * sr,) and np.isfinite(w).all() for w in waves.values())
        # One gain for every comparable sound in this case, never a per-method
        # match. Presentation criterion (codec fidelity, not a model judgement):
        # a candidate is a standalone preview when sharing a gain with the real
        # recording would leave the recording above the PCM16 tolerance, or when
        # the evaluator logged a scientific level-domain failure for it.
        reference_peak = float(np.abs(waves["reference"]).max())
        assert reference_peak > 0
        precision = {}
        for method in arms:
            pair_gain = 0.8 / max(reference_peak, float(np.abs(waves[method]).max()))
            error = pcm16_relative_error(waves["reference"], pair_gain)
            precision[method] = {
                "criterion": "pcm16_reference_roundtrip",
                "tolerance": PCM16_TOLERANCE,
                "pair_gain": pair_gain,
                "reference_relative_rms_error": error,
                "passes": error <= PCM16_TOLERANCE,
            }
        candidates = {
            m for m in arms if not precision[m]["passes"] or domains[m]["mics_out_of_domain"]
        }
        gain = 0.8 / max(float(np.abs(w).max()) for m, w in waves.items() if m not in candidates)
        gain_db = float(20 * np.log10(gain))
        playback = {
            m: gain if m not in candidates else min(gain, 0.8 / float(np.abs(waves[m]).max()))
            for m in waves
        }
        # Only an actual extra attenuation makes a card not level-comparable.
        excluded = {}
        for method in sorted(m for m in candidates if playback[m] < gain):
            reasons = []
            if not precision[method]["passes"]:
                reasons.append(
                    f"PCM16 presentation: at the gain shared with the recording "
                    f"({precision[method]['pair_gain']:.6g}) the recording's relative RMS coding error is "
                    f"{precision[method]['reference_relative_rms_error']:.4f} > {PCM16_TOLERANCE:g}"
                )
            if domains[method]["mics_out_of_domain"]:
                reasons.append(domains[method]["reason"])
            excluded[method] = "; ".join(reasons)
        members = [m for m in waves if m not in excluded]
        common_error = pcm16_relative_error(waves["reference"], gain)
        assert common_error <= PCM16_TOLERANCE, common_error
        figures, trajectory, spectral = listening_spectrograms(
            waves, rps, sr=sr, anchor="reference"
        )
        trajectory_path = out / f"{regime}__trajectory.png"
        trajectory.savefig(trajectory_path, dpi=180)
        plt.close(trajectory)
        note = (
            f" {len(excluded)} generated sound(s) would make the shared gain too small; each has its own "
            "attenuated preview and is marked NOT level-comparable."
            if excluded
            else ""
        )
        html.extend(
            [
                f'<section class="listening-case" data-case="{regime}" aria-labelledby="listen-{regime}">',
                f'<h3 id="listen-{regime}">{title}: {escape(key)}</h3>',
                f"<p>Saved draw 0; microphone 0 (first channel). Common playback gain: {gain_db:+.2f} dB. "
                f"This gain applies to the real recording and {len(members) - 1} of the {len(arms)} generated sounds "
                f"in this case.{note}</p>",
                '<div class="plot-scroll trajectory-plot">',
                f'<img src="assets/listening/{trajectory_path.name}" loading="lazy" '
                f'alt="The four recorded rotor-speed tracks for {escape(key)}. Every generator receives these tracks.">',
                "</div>",
                '<div class="listening-panels">',
            ]
        )
        case = {
            "key": key,
            "regime": regime,
            "draw": 0,
            "channel": 0,
            "sample_rate_hz": sr,
            "n_samples": 4 * sr,
            "start_s": pair["start_s"],
            "duration_s": 4.0,
            "trajectory_sha256": hashlib.sha256(np.ascontiguousarray(rps).tobytes()).hexdigest(),
            "window_provenance": window["provenance"],
            "worker_path_differences": {m: d for m, d in path_differences.items() if d},
            "playback_gain": gain,
            "playback_gain_db": gain_db,
            "pcm_peak_target": 0.8,
            "common_gain_rule": (
                "0.8 / max raw peak over all sounds except preview candidates; a candidate fails the PCM16 "
                "reference round-trip at its pair gain 0.8/max(reference peak, candidate peak), or has a "
                "logged generated level-domain failure; a candidate is excluded only if it then needs extra "
                "attenuation. No RMS matching, no clipping"
            ),
            "pcm16_reference_tolerance": PCM16_TOLERANCE,
            "pcm16_reference_rule": PCM16_RULE,
            "reference_relative_rms_error_at_common_gain": common_error,
            "common_gain_members": members,
            "common_gain_exclusions": excluded,
            "verified_neural_references": verified_refs,
            "spectrogram": spectral,
            "trajectory_image": str(trajectory_path.relative_to(HERE)),
            "methods": {},
        }
        for method, wave in waves.items():
            stem = f"{regime}__{method}"
            audio_path, image_path = out / f"{stem}.wav", out / f"{stem}.png"
            scaled = wave.astype(np.float64) * playback[method]
            assert np.abs(scaled).max() <= 0.8000001
            wavfile.write(audio_path, sr, np.rint(scaled * 32767).astype(np.int16))
            figures[method].savefig(image_path, dpi=180)
            plt.close(figures[method])
            kind = "real" if method == "reference" else "generated"
            label = labels[method]
            classes = f"listening-card {kind}" + (" failed-level" if method in excluded else "")
            html.extend(
                [
                    f'<article class="{classes}" data-method="{method}">',
                    f"<h4>{escape(label)}</h4>",
                ]
            )
            if method in excluded:
                extra_db = 20 * np.log10(gain / playback[method])
                why = []
                if not precision[method]["passes"]:
                    why.append(
                        "At a gain shared with the real recording, 16-bit coding would leave the recording with an "
                        f"RMS error of {100 * precision[method]['reference_relative_rms_error']:.1f} % "
                        f"(presentation tolerance {100 * PCM16_TOLERANCE:g} %)."
                    )
                if domains[method]["mics_out_of_domain"]:
                    why.append(
                        "Its scored level is outside the loudness-model domain (see the scoring note)."
                    )
                html.append(
                    '<p class="level-warning" role="note"><strong>NOT level-comparable.</strong> '
                    f"{' '.join(why)} This preview has an extra attenuation of {extra_db:.2f} dB relative to the "
                    "case gain. Do not compare its loudness with the other cards. The scored waveform is unchanged.</p>"
                )
            if method != "reference" and domains[method]["mics_out_of_domain"]:
                bad = domains[method]
                html.append(
                    '<p class="domain-warning" role="note"><strong>Scoring note.</strong> '
                    f"In this scored draw, {len(bad['mics_out_of_domain'])} of {len(bad['low_band_max_gen_db_per_mic'])} "
                    f"microphones exceed the {bad['limit_db']:g}-dB low-band limit (25–250 Hz) of the loudness model, "
                    f"at the fixed scoring gain (generated maximum {max(bad['low_band_max_gen_db_per_mic']):.1f} dB). "
                    "Its loudness and sharpness scores are undefined.</p>"
                )
            alt_extra = ""
            if method in spectral["display_saturated"]:
                alt_extra = (
                    f" Display saturated: the peak is {spectral['display_saturated'][method]['excess_db']:.1f} "
                    "dB above the colour-scale top."
                )
            html.extend(
                [
                    '<div class="plot-scroll">',
                    f'<img src="assets/listening/{image_path.name}" loading="lazy" '
                    f'alt="Spectrogram: {escape(label)}, {title.lower()}, microphone 0. '
                    f'The time, frequency and colour scales match the other methods in this case.{alt_extra}">',
                    "</div>",
                    f'<audio controls preload="metadata" aria-label="{escape(title)} — {escape(label)}">',
                    f'<source src="assets/listening/{audio_path.name}" type="audio/wav">',
                    f'<a href="assets/listening/{audio_path.name}">Download the WAV audio</a>',
                    "</audio>",
                ]
            )
            if method != "reference":
                links = [
                    f'<a href="#synth-{SYNTH_ANCHORS[method.removesuffix(REGIME_SUFFIX)]}">Synthesis equations</a>'
                ]
                if method in MATCHED:
                    links.append('<a href="#synth-two-experts">Two-expert blend</a>')
                html.append(f'<p class="equation-link">{" · ".join(links)}</p>')
            html.append("</article>")
            entry_gain = playback[method]
            case["methods"][method] = {
                "source": sources[method],
                "label": label,
                "audio": str(audio_path.relative_to(HERE)),
                "audio_sha256": digest(audio_path),
                "spectrogram_image": str(image_path.relative_to(HERE)),
                "raw_peak": float(np.abs(wave).max()),
                "raw_rms": float(np.sqrt(np.mean(wave.astype(np.float64) ** 2))),
                "playback_gain": entry_gain,
                "playback_gain_db": float(20 * np.log10(entry_gain)),
                "additional_attenuation_db": float(20 * np.log10(gain / entry_gain)),
                "level_comparable": method not in excluded,
                "level_reason": excluded.get(method, "common case gain"),
                "playback_precision": precision.get(method),
                "level_domain": domains.get(method),
                "display_saturated": spectral["display_saturated"].get(method),
            }
        html.extend(["</div>", "</section>", ""])
        manifest[regime] = case
    (assets / "listening.html").write_text("\n".join(html) + "\n")
    (out / "media.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def summarize_eone(fit: dict) -> tuple[str, list[str], dict]:
    """Fit-summary row, calibration table and provenance from the saved EONE fit report.

    Reads only fields written by ``eone_baseline.fit_eone`` and the frozen
    reload; a missing field fails instead of being guessed.
    """
    settings = fit["settings"]
    assert fit["method_id"] == "doerfler2026_eone" and fit["production_settings"] is True
    assert {k: settings[k] for k in EONE_PROSE_SETTINGS} == EONE_PROSE_SETTINGS
    assert fit["pretraining"]["used"] is False
    assert fit["torque"]["factor"] == 1.0 and fit["torque"]["learned"] is False
    assert fit["refitted_in_this_worker"] is False
    training = fit["training"]
    assert training["completed_epochs"] == training["planned_epochs"] == settings["epochs"]
    assert training["completed_steps"] == training["planned_steps"]
    n_mics, n_rotors, n_tones, n_bands, n_knots = (
        fit["n_mics"],
        fit["n_rotors"],
        fit["n_orders"],
        fit["n_bands"],
        fit["n_knots"],
    )
    n_values = fit["n_parameters"]
    assert n_values == n_mics * n_rotors * (n_tones + n_bands) * n_knots
    history = training["history"]
    best_epoch = training["best_epoch"]
    selected = history[best_epoch]
    assert (
        selected["epoch"] == best_epoch
        and selected["eval"]["total"] == training["best_eval_objective"]
    )
    assert history[0]["eval"]["total"] == training["initial_eval_objective"]
    assert history[-1]["eval"]["total"] == training["final_eval_objective"]
    pieces = fit["pieces"]
    manifest = pieces["manifest"]
    assert len(manifest) == pieces["n"]
    valid = sum(p["harmonic_tracks_valid"] for p in manifest)
    tracks = sum(p["harmonic_tracks_total"] for p in manifest)
    unvisited = fit["knot_coverage_frames"]["unvisited_knots_per_rotor"]
    assert len(unvisited) == n_rotors
    gate = fit["adapter_choices"]["alias_gate"]
    optimizer = fit["adapter_choices"]["optimizer"]
    speeds = fit["calibration_rps"]
    runtime, software = fit["runtime"], fit["software"]
    device = runtime["device"] + (
        f" ({software['device_name']})" if software["device_name"] else ""
    )
    selected_note = " (initialisation; no epoch lowered the objective)" if best_epoch == 0 else ""
    row = (
        f"| EONE-RPS curve bank | {pieces['n']} × {pieces['piece_s']:g} s | {n_values:,} positive gain-curve "
        "values; no pretrained prior | Multi-resolution STFT + tone-track energy + curvature penalty; "
        f"AdamW, {settings['epochs']} epochs; keep the best calibration epoch |"
    )
    lines = [
        "| Quantity | Saved value |",
        "|:--|:--|",
        f"| Calibration audio | {len(fit['calibration_clip_keys'])} windows of {', '.join(fit['calibration_recordings'])} "
        f"(split: {', '.join(fit['calibration_splits'])}), cut into {pieces['n']} non-overlapping "
        f"{pieces['piece_s']:g}-s pieces |",
        f"| Rotor speeds in these pieces | {speeds['min']:.1f}–{speeds['max']:.1f} rev/s; mean {speeds['mean']:.1f} rev/s |",
        f"| Curve values fitted | {n_values:,} = {n_mics} microphones × {n_rotors} rotors × "
        f"({n_tones} tone orders + {n_bands} noise bands) × {n_knots} speed knots |",
        f"| Knots that no calibration frame reaches, rotors 1–{n_rotors} | "
        f"{', '.join(str(int(u)) for u in unvisited)} of {n_knots} |",
        f"| Tone tracks that enter the harmonic loss | {valid:,} of {tracks:,} ({100 * valid / tracks:.1f} %) |",
        f"| Highest tone order with gate gain 1 at the top calibration speed, {speeds['max']:.1f} rev/s | "
        f"{gate['always_open_max_order_at_calibration_max_rps']} (pass edge {gate['pass_hz']:g} Hz) |",
        f"| Optimiser | AdamW; peak learning rate {settings['max_lr']:g}; weight decay {settings['weight_decay']:g}; "
        f"one-cycle cosine schedule; {optimizer['steps']:,} steps = {settings['epochs']} epochs × "
        f"{optimizer['steps_per_epoch']} pieces |",
        f"| Epochs completed | {training['completed_epochs']} of {training['planned_epochs']} |",
        f"| Objective at initialisation (epoch 0) | {training['initial_eval_objective']:.4f} |",
        f"| Selected epoch | {best_epoch}{selected_note} |",
        f"| Objective at the selected epoch | {training['best_eval_objective']:.4f} |",
        f"| Objective at the last epoch | {training['final_eval_objective']:.4f} |",
        f"| Relative reduction, initialisation to selected epoch | "
        f"{100 * training['relative_objective_improvement']:.1f} % |",
        f"| Selected-epoch parts: MRSTFT, harmonic, curvature | {selected['eval']['mrstft']:.4f}, "
        f"{selected['eval']['harmonic']:.4f}, {selected['eval']['curvature']:.3g} |",
        f"| Pretrained encoder–decoder | Not used: {fit['pretraining']['reason']} |",
        f"| Torque curve | Fixed factor {fit['torque']['factor']:g}; not learned |",
        f"| Fit device | {device}; training {runtime['train_seconds'] / 60:.1f} min; torch {software['torch']} |",
        f"| Frozen checkpoint | SHA-256 `{fit['checkpoint_file_sha256'][:16]}…`; reloaded, not refitted, for scoring |",
    ]
    record = {
        "checkpoint_file_sha256": fit["checkpoint_file_sha256"],
        "settings": settings,
        "pretraining": fit["pretraining"],
        "torque": fit["torque"],
        "training": {
            k: v for k, v in training.items() if k not in ("history", "step_total_loss", "step_lr")
        },
        "selected_epoch_eval": selected["eval"],
        "knot_coverage_unvisited_per_rotor": unvisited,
        "harmonic_tracks": {"valid": valid, "total": tracks},
        "runtime": runtime,
        "software": software,
    }
    return row, lines, record


def score_table(methods: tuple[str, ...], means: dict, counts: dict) -> list[str]:
    """Mean log-spectrum error; bold marks the lowest mean of each column within this table."""
    best = {r: min(means[m][r] for m in methods) for r in SCORE_REGIMES}
    lines = [
        "| Method | "
        + " | ".join(f"{REGIME_NAMES[r]} ({counts[r]})" for r in SCORE_REGIMES)
        + " |",
        "|:--|" + "--:|" * len(SCORE_REGIMES),
    ]
    for method in methods:
        cells = []
        for regime in SCORE_REGIMES:
            value = means[method][regime]
            cell = f"{value:.2f}"
            cells.append(f"**{cell}**" if value == best[regime] else cell)
        lines.append("| " + " | ".join([METHODS[method], *cells]) + " |")
    return lines


def result_summary(methods: tuple[str, ...], means: dict, *, compare_ours: bool = False) -> str:
    cells = []
    for r in SCORE_REGIMES:
        best = min(means[m][r] for m in methods)
        leaders = " and ".join(METHODS[m] for m in methods if means[m][r] == best)
        gap = means["stochastic"][r] - best
        relative = (
            f" Our model: {means['stochastic'][r]:.2f} dB." if compare_ours and gap > 0 else ""
        )
        cells.append(f"- **{REGIME_NAMES[r]}:** {leaders}, {best:.2f} dB.{relative}")
    return "\n".join(cells) + "\n"


def results_markdown(
    native: tuple[str, ...],
    primary: tuple[str, ...],
    pairs: dict[str, str],
    means: dict,
    counts: dict,
) -> str:
    paired = [
        "| Variant | "
        + " | ".join(
            f"{REGIME_NAMES[r]}: {t}" for r in SCORE_REGIMES for t in ("pooled", "two experts")
        )
        + " |",
        "|:--|" + "--:|" * (2 * len(SCORE_REGIMES)),
    ]
    for matched, pooled in pairs.items():
        cells = [f"{means[m][r]:.2f}" for r in SCORE_REGIMES for m in (pooled, matched)]
        paired.append("| " + " | ".join([POOLED[pooled], *cells]) + " |")
    lowered = {r: sum(means[m][r] < means[b][r] for m, b in pairs.items()) for r in REGIMES}
    return "\n".join(
        [
            "### Primary panel: same supports, same blend {#primary-panel}",
            "",
            f"Our model, its version without added tone variation, and {len(primary) - len(OURS)} two-expert arms. "
            "All the external arms in this panel use our standby and cruise supports and our power blend.",
            "",
            "::: {.table-scroll}",
            *score_table(primary, means, counts),
            ":::",
            "",
            "### Native pooled fits, retained {#pooled-panel}",
            "",
            "Each external variant here has one fit on the pooled windows (GP selects a subset). We repeat our model "
            "and its ablation for comparison. The bold cells apply only to this table.",
            "",
            "::: {.table-scroll}",
            *score_table(native, means, counts),
            ":::",
            "",
            "The lowest native-panel mean in each column:",
            "",
            result_summary(native, means),
            "### Same variant, two fitting treatments",
            "",
            "Only variants with both fits appear here. PTR, HPN and DDSP-Piano have no pooled fit. "
            "Each difference combines other fitting audio, a separate fit in each regime, and the blend. "
            "It does **not** show the effect of any one change by itself.",
            "",
            "::: {.table-scroll .fitting-table}",
            *paired,
            ":::",
            "",
            f"Of the {len(pairs)} paired variants, matching lowers {lowered['cruise']} means for fast rotors, "
            f"{lowered['standby']} for slow rotors and {lowered['transition']} for changing speeds.",
            "",
        ]
    )


def control_table(methods: tuple[str, ...], rows: dict) -> list[str]:
    lines = [
        "| Method | Tones found | Commanded rotor: tones found | Speed error given a match (rev/s) |",
        "|:--|--:|--:|--:|",
    ]
    lines.extend(f"| {METHODS[m]} | {rows[m][0]} | {rows[m][1]} | {rows[m][2]} |" for m in methods)
    return lines


def main() -> None:
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    evidence_path = ROOT / "results/noise_generator_paper/neural_extension_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    arms, native, primary, pairs = select_panels(evidence)
    matched = tuple(m for m in primary if m not in OURS)
    control_methods = tuple(m for m in arms if m != "stochastic_coherent")
    means = {}
    raws = {}
    provenance = {"evidence_sha256": digest(evidence_path), "arms": list(arms), "sources": {}}
    for method in arms:
        source = evidence["method_sources"][method]
        path = ROOT / source["raw_path"]
        assert digest(path) == source["sha256"], path
        raw = json.loads(path.read_text())
        assert raw["method"] == method and raw["schema"] == "generator-paper-evaluation/1", path
        assert raw["seed"] == 0 and raw["n_draws"] == 4
        groups = defaultdict(list)
        for pair in raw["pairs"]:
            groups[pair["regime"], pair["key"]].append(pair["summary"]["lsd_db"])
        assert len(groups) == 26 and all(len(v) == 4 for v in groups.values())
        means[method] = {}
        for regime in SCORE_REGIMES:
            values = [np.mean(v) for (r, _), v in groups.items() if r == regime or regime == "all"]
            expected = evidence["statistics"][method][regime]["metrics"]["lsd_db"]
            assert len(values) == expected["n"]
            assert np.isclose(np.mean(values), expected["mean"], rtol=0, atol=1e-12)
            means[method][regime] = float(np.mean(values))
        raws[method] = raw
        provenance["sources"][method] = source
    # Every arm is scored on the same evaluation windows.
    keys = [w["key"] for w in raws["stochastic"]["evaluation_windows"]]
    assert all([w["key"] for w in raws[m]["evaluation_windows"]] == keys for m in arms)
    counts = {
        r: evidence["statistics"]["stochastic"][r]["metrics"]["lsd_db"]["n"] for r in SCORE_REGIMES
    }

    supports, input_provenance_sha256 = core_supports()
    pooled_windows = raws["yang2025"]["calibration_windows"]
    assert all(
        [w["key"] for w in raws[m]["calibration_windows"]] == [w["key"] for w in pooled_windows]
        for m in native
        if m not in OURS
    )
    (assets / "regime-supports.md").write_text(support_table(supports, pooled_windows))
    provenance["regime_supports"] = {
        "input_provenance_sha256": input_provenance_sha256,
        "supports": supports,
        "pooled_window_keys": [w["key"] for w in pooled_windows],
    }
    provenance["matched"] = {m: check_matched(m, raws[m], supports) for m in matched}
    provenance["regime_comparison"] = {
        "primary_methods": primary,
        "pooled_panel_methods": native,
        "matched_methods": matched,
        "method_pairs": pairs,
    }
    (assets / "regime-fits.md").write_text(matched_fit_table(raws, matched))

    yang = raws["yang2025"]["calibration"]
    gp = raws["lee2026_gp"]["calibration"]
    gp_coloured = raws["lee2026_gp_colored"]["calibration"]
    assert gp["checkpoint_file_sha256"] == gp_coloured["checkpoint_file_sha256"]
    (assets / "reference-clip.md").unlink(missing_ok=True)
    n_yang = np.asarray(yang["offset_db"]).size + np.asarray(yang["slope_db_per_rpm"]).size
    n_candidates = len(yang["calibration_candidates"])
    n_frames = gp["frame_manifest"]["selected_frames"]
    frame_s = gp["settings"]["frame_s"]
    lines = [
        "| Method | Fitting audio | What is estimated or stored? | Fit criterion |",
        "|:--|:--|:--|:--|",
        "| PTR and HPN, two experts only | The standby or cruise supports; one fit per microphone | Four source models per microphone | Spectral and rotor-tone losses on the mixed waveform |",
        "| DDSP-Piano, two experts only | The standby or cruise supports; all microphones use one model | A four-voice synthesizer with a microphone environment input | The published spectral loss and tuning regularizers |",
        f"| Pulses, each clock | {yang['calibration_segments']} × 4 s | {n_yang} filter coefficients for each of {n_candidates} candidate settings | Band-level least squares, then corrected log-spectrum error |",
        f"| GP, both noise versions | {n_frames} × {frame_s:g} s; {gp['n_points']} microphone–frame points | {gp['n_tasks']} coefficient GPs; residual noise levels and shape | Waveform least squares, then GP marginal likelihood |",
        "| Our model | Earlier accepted FLY125 fits plus auxiliary measurements | Process parameters, spectral profiles and measured responses | Expected-spectrum loss plus priors; some groups measured or fixed |",
    ]
    eone_row, eone_lines, eone_record = summarize_eone(raws["doerfler2026_eone"]["calibration"])
    lines.insert(-1, eone_row)
    (assets / "fit-summary.md").write_text("\n".join(lines) + "\n")
    (assets / "eone-fit.md").write_text("\n".join(eone_lines) + "\n")
    lines = [
        "| Pulse clock | Pulse variation (Hz, peak-to-peak) | Noise amplitude ratio |",
        "|:--|--:|--:|",
    ]
    selected = {}
    for method, label in (("yang2025", "Integer periods"), ("yang2025_phase", "Fractional carry")):
        fit = raws[method]["calibration"]
        best = min(fit["calibration_candidates"], key=lambda c: c["calibration_lsd_db"])
        assert best["modulation_range_hz"] == fit["modulation_range_hz"]
        assert best["white_ratio"] == fit["white_ratio"]
        selected[method] = best
        lines.append(f"| {label} | {fit['modulation_range_hz']:g} | {fit['white_ratio']:g} |")
    (assets / "pulse-settings.md").write_text("\n".join(lines) + "\n")
    provenance["calibration"] = {
        "source": "calibration reports embedded in the hash-checked scored raw files",
        "selected_pulse_settings": selected,
        "gp_checkpoint_sha256": gp["checkpoint_file_sha256"],
        "gp_settings": gp["settings"],
        "proposed": raws["stochastic"]["calibration"],
        "eone": eone_record,
    }

    (assets / "results.md").write_text(results_markdown(native, primary, pairs, means, counts))
    (assets / "primary-summary.md").write_text(result_summary(primary, means, compare_ours=True))

    control_path = ROOT / evidence["control_root"] / "control_probe.json"
    assert digest(control_path) == evidence["control_source_sha256"]
    control = json.loads(control_path.read_text())
    assert control["methods"].keys() == evidence["control"].keys()
    assert set(control["methods"]) - ARCHIVED_METHODS == set(control_methods)
    rows = {}
    n_commands = 0
    for method in control_methods:
        if method in matched:
            report_hash = raws[method]["calibration"]["manifest_sha256"]
            control_meta = {k: v for k, v in control["methods"][method].items() if k != "records"}
            assert report_hash in json.dumps(control_meta), (
                f"{method}: control reused a different frozen fit"
            )
        rotors = [r for record in control["methods"][method]["records"] for r in record["rotors"]]
        errors = [
            abs(r["measured_bpf_hz"] / 2 - r["commanded_rps"]) for r in rotors if r["matched"]
        ]
        expected = evidence["control"][method]
        assert len(rotors) == expected["commands"] == 48
        assert len(errors) == expected["matches"]
        n_commands += len(rotors)
        if errors:
            error = float(np.mean(errors))
            assert np.isclose(error, expected["mae_given_match_rps"], rtol=0, atol=1e-12)
            cell = f"{error:.3f}"
        else:  # No matched tone: the conditional error is undefined.
            assert expected["mae_given_match_rps"] is None
            cell = "—"
        on = [r for r in rotors if r["rotor"] == 0]
        rows[method] = (
            f"{len(errors)}/{len(rotors)}",
            f"{sum(r['matched'] for r in on)}/{len(on)}",
            cell,
        )
    control_md = [
        "#### Primary panel: our model and the two-expert baselines",
        "",
        "::: {.table-scroll}",
        *control_table(("stochastic", *matched), rows),
        ":::",
        "",
        "#### Native pooled fits, retained",
        "",
        "::: {.table-scroll}",
        *control_table(tuple(m for m in native if m in rows), rows),
        ":::",
    ]
    (assets / "controls.md").write_text("\n".join(control_md) + "\n")
    provenance["control"] = {
        "path": str(control_path.relative_to(ROOT)),
        "sha256": digest(control_path),
        "methods": list(control_methods),
        "commands": n_commands,
    }

    # The saved figure cases were declared by the evaluator, before this explainer.
    raw = raws["stochastic"]
    first_standby = next(p["key"] for p in raw["pairs"] if p["regime"] == "standby")
    index = next(i for i, w in enumerate(raw["evaluation_windows"]) if w["key"] == first_standby)
    reference_path = ROOT / evidence["reference_examples_root"] / f"reference__{index:03d}.npz"
    with np.load(reference_path) as reference:
        assert str(reference["key"]) == first_standby and str(reference["regime"]) == "standby"
        waves = {"reference": reference["real"]}
        sr = int(reference["sr"])
    figure_sources = {
        "reference": {
            "path": str(reference_path.relative_to(ROOT)),
            "sha256": digest(reference_path),
        }
    }
    for method in OURS:
        path = (
            ROOT
            / evidence["method_sources"][method]["examples_root"]
            / f"{method}__{index:03d}.npz"
        )
        with np.load(path) as generated:
            assert str(generated["key"]) == first_standby and int(generated["sr"]) == sr
            waves[method] = generated["generated"]
        figure_sources[method] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
    assert all(w.shape == (8, 4 * sr) for w in waves.values())
    fig = tone_variation_spectrum(waves, sr=sr)
    fig.savefig(assets / "tone-variation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    provenance["figure"] = {"key": first_standby, "draw": 0, "sources": figure_sources}
    provenance["listening"] = listening = prepare_listening(assets, evidence, raws, arms)
    n_pairs = sum(len(raws[m]["pairs"]) for m in arms)
    n_clips = sum(len(case["methods"]) for case in listening.values())
    assert n_pairs == len(arms) * 26 * 4 and n_clips == len(REGIMES) * (len(arms) + 1)
    assert n_commands == 48 * len(control_methods)
    provenance["listening_players"] = n_clips
    provenance["level_previews"] = n_previews = sum(
        not entry["level_comparable"]
        for case in listening.values()
        for entry in case["methods"].values()
    )
    (assets / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        f"Verified {len(arms)} arms with {n_pairs} saved comparison pairs and {len(control_methods)} "
        f"control methods with {n_commands} commands; exported {n_clips} matched audio clips and spectrograms "
        f"({n_previews} not level-comparable previews)."
    )


if __name__ == "__main__":
    main()
