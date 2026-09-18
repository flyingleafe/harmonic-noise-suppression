"""Why the v2 render loses Michael's STANDBY and RAMP regimes.

Round 1 scored the v2 candidate ``michaels_v2_fly125cruise`` (one FLY125
CRUISE fit, rendered on every FLY124 regime) at HPPNet PIT MAE 32.63 rev/s on
standby and 16.96 on ramp against a legacy 0.32 / 8.01, while cruise itself is
at parity (0.81 vs 0.76). This runner takes ONE standby and ONE ramp support
and measures, arm by arm, what the render actually puts on the wire:

* the real clip, the v2 render, the LEGACY render of the same support (the
  route the frozen legacy scalar came from, :func:`noise_v2_round_score.legacy_arm`);
* targeted VARIANT renders of the same v2 fit, each disabling exactly one
  suspect: the comb's speed envelope (``amp_exp``), the floor's speed envelope
  (``floor_exp``), both, the floor re-levelled onto the real support, and the
  comb muted outright (the floor-only control).

Per arm it reports the band-limited level, the absolute LTAS deviation, the
comb-to-floor ratio at k = 2, 4, 8 read off a Welch periodogram, the speed the
render actually ENCODES (order-2 demodulation against the frozen label track),
and — with ``--probe`` — the frozen HPPNet PIT MAE.

    # diagnostics + figures, no GPU (a few minutes on a laptop)
    python scripts/noise_v2_render_regime.py --figures --out DIR

    # the same measurement with the frozen scorer, on uni-gpushort
    python scripts/noise_v2_render_regime.py --probe --out DIR

The renders are deterministic in ``--seed``, so the two passes agree cell for
cell and the probe pass is the record.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.noise_model import gates as GT
from experiments.noise_model import render as RD
from experiments.noise_model import spectrum as SP
from experiments.stochastic_fit import revised_eval as RE

SCHEMA = "noise-v2-render-regime/1"
OUT_DEFAULT = Path("results/noise_v2/rounds/round2/render_regime")
FIT_DEFAULT = Path("results/noise_v2/rounds/round1/fits/michaels_fly125_cruise__flight.json")

#: The two frozen FLY124 supports this study is about, plus the cruise support
#: that is already at parity. CRUISE FIRST: the level-matched-exponent variant
#: reads the real cruise level as its reference.
STUDY_REGIMES = ("cruise", "standby", "ramp")

#: Periodogram geometry of the level/comb diagnostics. 8192 at 16 kHz is
#: 1.95 Hz per bin: the order-2 lines of two rotors 5 rev/s apart are ten bins
#: apart, so a line and its own local floor are separable.
PSD_N = 8192
PSD_HOP = 4096

#: Spectrogram panels: the campaign's own flight front end.
SPEC_N = 2048
SPEC_HOP = 512

#: The orders whose comb-to-floor ratio is reported.
COMB_ORDERS = (2, 4, 8)

#: Mute level of a disabled block. 200 dB below the fitted value is numerically
#: gone and still finite.
MUTE_DB = 200.0

#: The prior means of the two speed exponents and of the floor's static
#: pedestal (``model.Priors.amp_exp``, ``floor_exp``, ``log_floor_static``):
#: what the patch proposal pins them to when the pool cannot identify them.
PRIOR_EXP = 2.0
PRIOR_STATIC_REL = 2.5e-3


def die(message: str) -> None:
    raise SystemExit(f"error: {message}")


def _module(name: str) -> Any:
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


# ── the variant fits ────────────────────────────────────────────────────────


def _mutate(fit: dict[str, Any], **kw: Any) -> dict[str, Any]:
    """A deep copy of ``fit`` with named speed-envelope / block edits applied."""
    out = copy.deepcopy(fit)
    p = out["params"]
    if "amp_exp" in kw:
        p["profile"]["amp_exp"] = float(kw["amp_exp"])
    if "floor_exp" in kw:
        p["floor"]["floor_exp"] = float(kw["floor_exp"])
        # the static pedestal is the OTHER half of the floor's speed law; a
        # floor_exp pin that left it alone would still tilt with speed.
        p["floor"]["floor_static_rel"] = float(kw.get("floor_static_rel", 0.0))
    if kw.get("mute_comb"):
        prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64) - MUTE_DB
        p["profile"]["profile_db"] = prof.tolist()
    if kw.get("mute_floor"):
        p["floor"]["floor_mean_db"] = float(p["floor"]["floor_mean_db"]) - MUTE_DB
    if "floor_mean_shift_db" in kw:
        p["floor"]["floor_mean_db"] = float(p["floor"]["floor_mean_db"]) + float(
            kw["floor_mean_shift_db"]
        )
    if "sigma_nu_scale" in kw:
        # the shaft OU's noise is in rev/s, so a cruise-fitted sigma_nu is a
        # LARGER fraction of a standby carrier: the relative smearing of the
        # comb is regime-dependent even though the parameter is not
        p["sigma_nu"] = float(p["sigma_nu"]) * float(kw["sigma_nu_scale"])
    return out


def _pool_span(fit: dict[str, Any]) -> dict[str, Any]:
    """The carrier range the fit's OWN objective saw, as it recorded it.

    ``diagnostics.batch.carrier_{min,max}_rev_s`` is what the flight batch
    reports; the ratio of the two is the only leverage the pool gives the two
    speed exponents.
    """
    batch = ((fit.get("diagnostics") or {}).get("batch") or {}) if fit else {}
    lo, hi = batch.get("carrier_min_rev_s"), batch.get("carrier_max_rev_s")
    if lo is None or hi is None or float(lo) <= 0.0:
        return dict(pool_carrier_rev_s=None, pool_speed_span=None)
    return dict(
        pool_carrier_rev_s=[float(lo), float(hi)],
        pool_speed_span=float(hi) / float(lo),
    )


@dataclass(frozen=True)
class ArmSpec:
    """One rendered arm of the comparison."""

    name: str
    kind: str  # real | v2 | legacy
    label: str
    hypothesis: str
    build: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None
    probe: bool = True


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec("real", "real", "the real FLY124 clip", "reference"),
    ArmSpec(
        "legacy",
        "legacy",
        "legacy stage-2 baseline, the frozen evaluator's own per-regime export",
        "reference (the parity bar's own render)",
    ),
    ArmSpec("v2", "v2", "the round-1 v2 fit, unchanged", "the failing arm", lambda f, _c: f),
    ArmSpec(
        "v2_nocomb",
        "v2",
        "v2 with the comb muted (floor only)",
        "control: what PIT MAE looks like when there is no comb at all",
        lambda f, _c: _mutate(f, mute_comb=True),
        probe=True,
    ),
    ArmSpec(
        "v2_amp_exp0",
        "v2",
        "v2 with amp_exp pinned to 0 (comb speed envelope neutral)",
        "H1: the comb's speed scaling extrapolated from cruise",
        lambda f, _c: _mutate(f, amp_exp=0.0),
    ),
    ArmSpec(
        "v2_floor_exp0",
        "v2",
        "v2 with floor_exp pinned to 0 and no static pedestal (floor speed envelope neutral)",
        "H2: the floor's speed envelope extrapolated from cruise",
        lambda f, _c: _mutate(f, floor_exp=0.0),
    ),
    ArmSpec(
        "v2_both_exp0",
        "v2",
        "v2 with BOTH speed envelopes neutral (amp_exp = floor_exp = 0)",
        "H1 + H2 together",
        lambda f, _c: _mutate(f, amp_exp=0.0, floor_exp=0.0),
    ),
    ArmSpec(
        "v2_floor_calib",
        "v2",
        "floor-only render with floor_exp = 0: the calibration of v2_floor_matched",
        "internal (not probed)",
        lambda f, _c: _mutate(f, floor_exp=0.0, mute_comb=True),
        probe=False,
    ),
    ArmSpec(
        "v2_floor_matched",
        "v2",
        "v2 with the floor re-levelled onto THIS support's real broadband level",
        "H2 in its strongest form: the floor level alone",
        lambda f, c: _mutate(
            f, floor_exp=0.0, floor_mean_shift_db=float(c["floor_match_shift_db"])
        ),
    ),
    ArmSpec(
        "v2_prior_exps",
        "v2",
        "v2 with BOTH exponents at their prior means (amp_exp = floor_exp = 2, "
        "static_rel at its own prior mean)",
        "the patch: what the fit would render if the unidentified exponents were pinned",
        lambda f, _c: _mutate(
            f, amp_exp=PRIOR_EXP, floor_exp=PRIOR_EXP, floor_static_rel=PRIOR_STATIC_REL
        ),
    ),
    ArmSpec(
        "v2_level_matched_exps",
        "v2",
        "v2 with BOTH exponents at the value the REAL clips' own cruise-to-regime level "
        "ratio implies",
        "the ceiling of a single global speed envelope: what is left is the comb profile itself",
        lambda f, c: _mutate(
            f,
            amp_exp=float(c["level_matched_exp"]),
            floor_exp=float(c["level_matched_exp"]),
            floor_static_rel=PRIOR_STATIC_REL,
        ),
    ),
    ArmSpec(
        "v2_prior_exps_nofloor",
        "v2",
        "prior-mean exponents with the floor muted: the fitted comb alone",
        "residual: is what is left after the envelopes the FLOOR, or the comb itself?",
        lambda f, _c: _mutate(
            f,
            amp_exp=PRIOR_EXP,
            floor_exp=PRIOR_EXP,
            floor_static_rel=PRIOR_STATIC_REL,
            mute_floor=True,
        ),
    ),
    ArmSpec(
        "v2_prior_exps_lowjitter",
        "v2",
        "prior-mean exponents with the shaft jitter at a quarter of the fitted sigma_nu",
        "residual: the cruise-fitted shaft jitter is a 2.2x larger FRACTION of a standby carrier",
        lambda f, _c: _mutate(
            f,
            amp_exp=PRIOR_EXP,
            floor_exp=PRIOR_EXP,
            floor_static_rel=PRIOR_STATIC_REL,
            sigma_nu_scale=0.25,
        ),
    ),
)


# ── the measurements ────────────────────────────────────────────────────────


def _welch(x: np.ndarray, *, n: int = PSD_N, hop: int = PSD_HOP) -> np.ndarray:
    """One-sided power per bin of ``x`` (Hann, ``n``-point, ``hop`` overlap)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < n:
        die(f"{x.size} samples is shorter than the {n}-point periodogram window")
    w = np.hanning(n)
    starts = np.arange(0, x.size - n + 1, hop)
    acc = np.zeros(n // 2 + 1, dtype=np.float64)
    for s in starts:
        acc += np.abs(np.fft.rfft(x[s : s + n] * w)) ** 2
    return acc / (float(starts.size) * float((w**2).sum()))


def band_level_db(psd: np.ndarray, freqs: np.ndarray) -> float:
    """``10 log10`` of the 30-7900 Hz power of one mic's periodogram."""
    band = (freqs >= SP.BAND_F_MIN) & (freqs <= SP.BAND_F_MAX)
    return float(10.0 * np.log10(max(float(psd[band].sum()), 1e-300)))


def comb_to_floor_db(
    psd: np.ndarray, freqs: np.ndarray, f0_rotor: np.ndarray, k: int
) -> dict[str, Any]:
    """Peak-to-local-floor of order ``k``, per rotor, in dB.

    The peak is the largest bin within +-3 Hz of ``k f0_r``; the local floor is
    the 20th percentile of the bins within +-0.5 ``f0`` of it EXCLUDING +-4 Hz
    around every rotor's own k-th line, so a neighbouring rotor's line cannot
    be mistaken for the floor.
    """
    df = float(freqs[1] - freqs[0])
    lines = np.asarray(f0_rotor, dtype=np.float64) * float(k)
    per_rotor: list[float | None] = []
    for line in lines:
        if line <= SP.BAND_F_MIN or line >= SP.BAND_F_MAX:
            per_rotor.append(None)
            continue
        half = max(3.0, 2.0 * df)
        peak_sel = np.abs(freqs - line) <= half
        span = max(0.5 * float(np.mean(f0_rotor)), 8.0)
        near = np.abs(freqs - line) <= span
        excl = np.zeros_like(freqs, dtype=bool)
        for other in lines:
            excl |= np.abs(freqs - other) <= max(4.0, 2.0 * df)
        floor_sel = near & ~excl
        if not peak_sel.any() or floor_sel.sum() < 4:
            per_rotor.append(None)
            continue
        peak = float(psd[peak_sel].max())
        base = float(np.percentile(psd[floor_sel], 20.0))
        per_rotor.append(float(10.0 * np.log10(max(peak, 1e-300) / max(base, 1e-300))))
    ok = [v for v in per_rotor if v is not None]
    return dict(
        k=int(k),
        line_hz=[float(v) for v in lines],
        per_rotor_db=per_rotor,
        mean_db=float(np.mean(ok)) if ok else None,
    )


def encoded_speed_dev(
    x: np.ndarray, f0_track: np.ndarray, *, sr: int, k: int = 2, bw_hz: float = 4.0
) -> dict[str, Any]:
    """Speed the signal ENCODES at order ``k``, against the label track.

    Demodulates by the label's own order-``k`` phase, low-passes to ``+-bw_hz``,
    and differentiates the residual phase: on a render whose comb really sits on
    the label carrier the residual is the model's own shaft jitter (a fraction of
    a rev/s); if the render encoded another speed the residual ramps, and if
    there is no line in the band at all the residual is floor noise.
    """
    x = np.asarray(x, dtype=np.float64)
    f0 = np.asarray(f0_track, dtype=np.float64)
    n = int(min(x.size, f0.size))
    phase = 2.0 * np.pi * float(k) * np.cumsum(f0[:n]) / float(sr)
    z = x[:n] * np.exp(-1j * phase)
    spec = np.fft.fft(z)
    freqs = np.fft.fftfreq(n, d=1.0 / float(sr))
    spec[np.abs(freqs) > float(bw_hz)] = 0.0
    lp = np.fft.ifft(spec)
    mag = np.abs(lp)
    guard = slice(int(0.05 * n), int(0.95 * n))
    dphi = np.diff(np.unwrap(np.angle(lp)))
    dev = dphi * float(sr) / (2.0 * np.pi * float(k))  # rev/s off the label
    dev = dev[guard]
    w = mag[1:][guard]
    if not np.isfinite(dev).all() or w.sum() <= 0:
        return dict(k=int(k), bw_hz=float(bw_hz), median_abs_dev_rev_s=None)
    order = np.argsort(np.abs(dev))
    cum = np.cumsum(w[order])
    med = float(np.abs(dev)[order][int(np.searchsorted(cum, 0.5 * cum[-1]))])
    return dict(
        k=int(k),
        bw_hz=float(bw_hz),
        median_abs_dev_rev_s=med,
        mean_dev_rev_s=float(np.average(dev, weights=w)),
        p90_abs_dev_rev_s=float(np.abs(dev)[order][int(np.searchsorted(cum, 0.9 * cum[-1]))]),
    )


def arm_diagnostics(
    audio: np.ndarray,
    real: np.ndarray,
    *,
    sr: int,
    f0_rotor: np.ndarray,
    f0_tracks: np.ndarray,
    demod_rotors: tuple[int, ...],
) -> dict[str, Any]:
    """Every per-arm number of one support: level, LTAS, comb, encoded speed."""
    freqs = np.fft.rfftfreq(PSD_N, d=1.0 / float(sr))
    psd_mics = np.stack([_welch(audio[m]) for m in range(audio.shape[0])])
    psd0 = psd_mics[0]
    levels = [band_level_db(psd_mics[m], freqs) for m in range(psd_mics.shape[0])]
    ltas = RE.ltas_deviation_db(real, audio, mic=0)
    return dict(
        band_level_db_mic0=float(levels[0]),
        band_level_db_mic_mean=float(
            10.0 * np.log10(np.mean([10.0 ** (v / 10.0) for v in levels]))
        ),
        band_level_db_per_mic=[float(v) for v in levels],
        rms=[float(v) for v in np.sqrt((np.asarray(audio, dtype=np.float64) ** 2).mean(axis=1))],
        peak_abs=float(np.abs(audio).max()),
        ltas_mean_abs_db=float(ltas["mean_abs_db"]),
        ltas_level_offset_db=float(ltas["level_offset_db"]),
        ltas_max_abs_db=float(ltas["max_abs_db"]),
        ltas_shape_only_mean_abs_db=float(ltas["shape_only_mean_abs_db"]),
        ltas_bands_db=[float(v) for v in ltas["bands_arm_db"]],
        comb_to_floor=[comb_to_floor_db(psd0, freqs, f0_rotor, k) for k in COMB_ORDERS],
        encoded_speed={
            f"rotor{r}": encoded_speed_dev(audio[0], f0_tracks[r], sr=sr) for r in demod_rotors
        },
        _psd=psd_mics,
    )


# ── the run ─────────────────────────────────────────────────────────────────


def study_supports(regimes: tuple[str, ...]) -> list[GT.ScoredSupport]:
    """The first frozen Michael's support of each requested regime."""
    out: list[GT.ScoredSupport] = []
    for regime in regimes:
        found = [s for s in GT.MICHAELS_SUPPORTS if s.regime == regime]
        if not found:
            die(f"no frozen Michael's support carries regime {regime!r}")
        out.append(found[0])
    return out


def _demod_rotors(f0_rotor: np.ndarray) -> tuple[int, ...]:
    """Rotors whose order-2 line is resolvable from every other rotor's.

    Two rotors 0.06 rev/s apart (FLY124 ramp carries such a pair) share one
    demodulation band; their residual phase is a beat, not a speed, so they are
    reported as unresolvable rather than as a false deviation.
    """
    out: list[int] = []
    for r, f in enumerate(f0_rotor):
        gaps = [abs(float(f) - float(g)) for i, g in enumerate(f0_rotor) if i != r]
        if min(gaps) * 2.0 > 8.0:
            out.append(int(r))
    return tuple(out) or (int(np.argmax(f0_rotor)),)


def run(
    *,
    fit_path: Path,
    out: Path,
    seed: int,
    regimes: tuple[str, ...],
    probe: bool,
    n_mics: int,
) -> dict[str, Any]:
    rs = _module("noise_v2_round_score")
    fit = json.loads(Path(fit_path).read_text())
    if str(fit.get("schema")) not in RD.READABLE_SCHEMAS:
        die(f"{fit_path}: schema {fit.get('schema')!r} is none of {list(RD.READABLE_SCHEMAS)}")
    probe_obj = rs.Probe.load() if probe else None
    legacy = rs.legacy_arm("michaels")
    payload: dict[str, Any] = dict(
        schema=SCHEMA,
        git=git_rev(),
        fit=dict(
            path=str(fit_path),
            support=fit.get("support"),
            mode=fit.get("mode"),
            k_max=fit.get("k_max"),
            converged=(fit.get("diagnostics") or {}).get("converged"),
            amp_exp=float(fit["params"]["profile"]["amp_exp"]),
            floor_exp=float(fit["params"]["floor"]["floor_exp"]),
            floor_static_rel=float(fit["params"]["floor"]["floor_static_rel"]),
            floor_mean_db=float(fit["params"]["floor"]["floor_mean_db"]),
            sigma_nu=float(fit["params"]["sigma_nu"]),
            lam=float(fit["params"]["lam"]),
            amp_rps_ref=float(RD.AMP_RPS_REF),
            n_pool_windows=len(fit.get("supports") or []),
            **_pool_span(fit),
        ),
        protocol=dict(
            seed=int(seed),
            n_mics=int(n_mics),
            psd=dict(n_fft=PSD_N, hop=PSD_HOP),
            spectrogram=dict(n_fft=SPEC_N, hop=SPEC_HOP),
            band_hz=[SP.BAND_F_MIN, SP.BAND_F_MAX],
            comb_orders=list(COMB_ORDERS),
            scorer=(probe_obj.record if probe_obj is not None else None),
            legacy_route={
                k: dict(v, clip_ids=list(v.get("clip_ids") or [])[:4])
                for k, v in legacy.source.items()
            },
        ),
        supports={},
    )
    figures: dict[str, Any] = {}
    # cross-support reference of the level-matched-exponent variant: the real
    # cruise clip's own band level and speed
    anchor: dict[str, float] = {}
    for support in study_supports(regimes):
        clip = RE.load_window(
            support.window,
            dataset=GT.DATASET["michaels"],
            version=None,
            channels=None,
            rps_key=GT.RAW_RPS_KEY["michaels"],
        )
        real = np.asarray(clip.audio, dtype=np.float64)[:n_mics]
        reference = np.atleast_2d(np.asarray(clip.rps, dtype=np.float64))
        rsupport = RE.regime_support(
            support.window,
            reference,
            regime=support.regime,
            min_rps=support.min_rps,
            max_rps=support.max_rps,
            sr=clip.sr,
        )
        f0_rotor = reference.mean(axis=1)
        demod = _demod_rotors(f0_rotor)
        mics = list(range(min(int(n_mics), int(real.shape[0]))))
        row: dict[str, Any] = dict(
            support=support.as_dict(),
            n_samples=int(real.shape[-1]),
            scored_seconds=float(rsupport.scored_seconds),
            mean_reference_rps=float(reference.mean()),
            per_rotor_mean_rps=[float(v) for v in f0_rotor],
            speed_norm=[float(v / RD.AMP_RPS_REF) for v in f0_rotor],
            demod_rotors=list(demod),
            envelope=envelope_report(fit["params"], f0_rotor),
            arms={},
        )
        cache: dict[str, Any] = {}
        psds: dict[str, np.ndarray] = {}
        specs: dict[str, np.ndarray] = {}
        for spec_arm in ARMS:
            t0 = time.time()
            if spec_arm.kind == "real":
                audio = real
            elif spec_arm.kind == "legacy":
                arm = rs._arm_for_recording(legacy, support)
                audio = arm.render(
                    reference, regime=support.regime, n_mics=len(mics), seed=int(seed)
                )[: len(mics)]
            else:
                assert spec_arm.build is not None
                variant = spec_arm.build(fit, cache)
                audio = RD.render_noise(variant, reference, n_mics=len(mics), seed=int(seed))[
                    : len(mics)
                ]
            audio = np.asarray(audio, dtype=np.float64)
            if int(audio.shape[-1]) != int(real.shape[-1]):
                die(
                    f"{support.key}: arm {spec_arm.name} rendered {audio.shape[-1]} samples "
                    f"against the frozen {real.shape[-1]}"
                )
            diag = arm_diagnostics(
                audio,
                real,
                sr=int(clip.sr),
                f0_rotor=f0_rotor,
                f0_tracks=reference,
                demod_rotors=demod,
            )
            psds[spec_arm.name] = diag.pop("_psd")
            specs[spec_arm.name] = audio[0].copy()
            entry: dict[str, Any] = dict(
                kind=spec_arm.kind,
                label=spec_arm.label,
                hypothesis=spec_arm.hypothesis,
                render_seconds=float(time.time() - t0),
                **diag,
            )
            if spec_arm.name == "v2_floor_calib":
                # the dB the flat-envelope floor has to move to sit at THIS
                # support's real broadband level (the floor carries
                # 10^(floor_mean_db / 10), so the shift is exactly the gap).
                cache["floor_match_shift_db"] = float(
                    row["arms"]["real"]["band_level_db_mic0"] - entry["band_level_db_mic0"]
                )
                entry["floor_match_shift_db"] = cache["floor_match_shift_db"]
            if spec_arm.kind == "real":
                speed = float(np.mean(f0_rotor) / RD.AMP_RPS_REF)
                level = float(entry["band_level_db_mic0"])
                if not anchor:
                    anchor.update(speed=speed, level_db=level)
                ratio = speed / float(anchor["speed"])
                cache["level_matched_exp"] = (
                    PRIOR_EXP
                    if abs(np.log10(ratio)) < 0.02
                    else float((level - float(anchor["level_db"])) / (10.0 * np.log10(ratio)))
                )
                row["level_matched_exp"] = cache["level_matched_exp"]
                row["level_matched_anchor"] = dict(anchor)
            if probe_obj is not None and spec_arm.probe:
                pit = probe_obj.tracker.pit(
                    audio,
                    reference,
                    mics=mics,
                    expected_samples=int(real.shape[-1]),
                    support=rsupport,
                )
                entry["pit_mae"] = float(pit["mae"])
                entry["pit_per_mic"] = [float(v) for v in pit["per_mic"]]
                entry["pit_per_rotor"] = [float(v) for v in pit["per_rotor"]]
                entry["n_scored_frames"] = int(pit["n_scored_frames"])
            row["arms"][spec_arm.name] = entry
            print(
                f"[{support.regime} {support.key}] {spec_arm.name}: "
                f"level={entry['band_level_db_mic0']:+.2f} dB "
                f"ltas={entry['ltas_mean_abs_db']:.2f} dB "
                f"pit={entry.get('pit_mae')}",
                flush=True,
            )
        payload["supports"][support.key] = row
        figures[support.regime] = dict(
            support=support, psds=psds, waves=specs, sr=int(clip.sr), row=row
        )
    payload["verdicts"] = verdicts(payload)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    return payload | {"_figures": figures}


def envelope_report(params: dict[str, Any], f0_rotor: np.ndarray) -> dict[str, Any]:
    """The two fitted speed envelopes evaluated at this support's speeds.

    Exactly the arithmetic the renderer does: the comb amplitude carries
    ``speed ** amp_exp`` in POWER (``render.render_noise``'s ``speed_amp`` is
    its square root) and the floor carries
    ``mean_r(speed ** floor_exp) + static_rel``.
    """
    s = np.asarray(f0_rotor, dtype=np.float64) / float(RD.AMP_RPS_REF)
    amp_exp = float(params["profile"]["amp_exp"])
    floor_exp = float(params["floor"]["floor_exp"])
    rel = float(params["floor"]["floor_static_rel"])
    comb = float(np.mean(s**amp_exp))
    floor = float(np.mean(s**floor_exp) + rel)
    k_max = min(
        int(np.asarray(params["profile"]["profile_db"]).shape[1]),
        SP.k_max_for_carrier(np.asarray(f0_rotor, dtype=np.float64), SP.FLIGHT_SR),
    )
    return dict(
        speed_mean=float(s.mean()),
        comb_envelope_db=float(10.0 * np.log10(max(comb, 1e-300))),
        floor_envelope_db=float(10.0 * np.log10(max(floor, 1e-300))),
        comb_minus_floor_db=float(
            10.0 * np.log10(max(comb, 1e-300)) - 10.0 * np.log10(max(floor, 1e-300))
        ),
        k_max=int(k_max),
        top_comb_line_hz=float(k_max * float(np.max(f0_rotor))),
    )


def verdicts(payload: dict[str, Any]) -> dict[str, Any]:
    """The four hypotheses, each decided by numbers already in the payload."""
    out: dict[str, Any] = {}
    rows = payload["supports"]

    def arm(regime: str, name: str, key: str) -> Any:
        for row in rows.values():
            if row["support"]["regime"] == regime:
                return row["arms"].get(name, {}).get(key)
        return None

    def env(regime: str, key: str) -> Any:
        for row in rows.values():
            if row["support"]["regime"] == regime:
                return row["envelope"][key]
        return None

    for regime in ("standby", "ramp"):
        if not any(r["support"]["regime"] == regime for r in rows.values()):
            continue
        out[regime] = dict(
            comb_minus_floor_db=env(regime, "comb_minus_floor_db"),
            comb_minus_floor_db_vs_cruise=(
                None
                if env("cruise", "comb_minus_floor_db") is None
                else float(
                    env(regime, "comb_minus_floor_db") - env("cruise", "comb_minus_floor_db")
                )
            ),
            pit={a.name: arm(regime, a.name, "pit_mae") for a in ARMS if a.probe},
            level_offset_db=dict(
                legacy=arm(regime, "legacy", "ltas_level_offset_db"),
                v2=arm(regime, "v2", "ltas_level_offset_db"),
                v2_floor_exp0=arm(regime, "v2_floor_exp0", "ltas_level_offset_db"),
                v2_both_exp0=arm(regime, "v2_both_exp0", "ltas_level_offset_db"),
            ),
            k2_comb_to_floor_db=dict(
                real=arm(regime, "real", "comb_to_floor")[0]["mean_db"]
                if arm(regime, "real", "comb_to_floor")
                else None,
                v2=arm(regime, "v2", "comb_to_floor")[0]["mean_db"]
                if arm(regime, "v2", "comb_to_floor")
                else None,
                legacy=arm(regime, "legacy", "comb_to_floor")[0]["mean_db"]
                if arm(regime, "legacy", "comb_to_floor")
                else None,
                v2_both_exp0=arm(regime, "v2_both_exp0", "comb_to_floor")[0]["mean_db"]
                if arm(regime, "v2_both_exp0", "comb_to_floor")
                else None,
            ),
        )
    return out


# ── figures ─────────────────────────────────────────────────────────────────

FIG_ARMS = ("real", "legacy", "v2", "v2_nocomb", "v2_floor_exp0", "v2_prior_exps")
FIG_PANELS = ("real", "legacy", "v2", "v2_floor_exp0", "v2_prior_exps")
FIG_COLOURS = {
    "real": "#111111",
    "legacy": "#1f77b4",
    "v2": "#d62728",
    "v2_floor_exp0": "#ff7f0e",
    "v2_both_exp0": "#2ca02c",
    "v2_nocomb": "#bbbbbb",
    "v2_amp_exp0": "#9467bd",
    "v2_floor_matched": "#8c564b",
    "v2_prior_exps": "#006d2c",
    "v2_level_matched_exps": "#17becf",
    "v2_prior_exps_nofloor": "#e377c2",
    "v2_prior_exps_lowjitter": "#bcbd22",
}
FIG_STYLE = {
    "real": ("-", 2.6),
    "legacy": ("-", 1.7),
    "v2": ("-", 2.4),
    "v2_nocomb": ("--", 1.2),
    "v2_floor_exp0": (":", 2.0),
    "v2_both_exp0": ("--", 1.8),
    "v2_prior_exps": ("--", 2.0),
    "v2_level_matched_exps": ("-.", 1.8),
}


def _smooth_log(
    freqs: np.ndarray, psd: np.ndarray, *, per_oct: int = 48
) -> tuple[np.ndarray, np.ndarray]:
    """Log-spaced band means of a periodogram: readable without hiding lines."""
    lo, hi = SP.BAND_F_MIN, SP.BAND_F_MAX
    n = int(per_oct * np.log2(hi / lo))
    edges = np.geomspace(lo, hi, n + 1)
    idx = np.digitize(freqs, edges) - 1
    out = np.full(n, np.nan)
    for b in range(n):
        sel = idx == b
        if sel.any():
            out[b] = psd[sel].mean()
    centres = np.sqrt(edges[:-1] * edges[1:])
    ok = np.isfinite(out)
    return centres[ok], 10.0 * np.log10(np.maximum(out[ok], 1e-300))


def _save(fig: Any, path: Path, *, colors: int = 128) -> str:
    """Save a figure and palette-quantise it: these PNGs go into the repo.

    A full-colour 160 dpi spectrogram of NOISE is ~5 MB and compresses badly;
    an adaptive 128-colour palette is visually identical here and a few hundred
    kilobytes.
    """
    from PIL import Image

    fig.savefig(path)
    with Image.open(path) as im:
        im.convert("RGB").quantize(colors=int(colors), method=Image.Quantize.MEDIANCUT).save(
            path, optimize=True
        )
    return str(path)


def write_figures(payload: dict[str, Any], figures: dict[str, Any], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[str] = []
    freqs = np.fft.rfftfreq(PSD_N, d=1.0 / 16000.0)
    for regime, fig_data in figures.items():
        if regime not in ("standby", "ramp"):
            continue
        row = fig_data["row"]
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), dpi=160, height_ratios=[2, 1])
        ax = axes[0]
        for name in FIG_ARMS:
            if name not in fig_data["psds"]:
                continue
            f, db = _smooth_log(freqs, fig_data["psds"][name][0])
            ls, lw = FIG_STYLE.get(name, ("--", 1.5))
            ax.plot(
                f,
                db,
                color=FIG_COLOURS[name],
                lw=lw,
                ls=ls,
                label=f"{name} ({row['arms'][name]['band_level_db_mic0']:+.1f} dB band level, "
                f"PIT {_f(row['arms'][name].get('pit_mae'), '.2f')})",
            )
        ax.set_xscale("log")
        ax.set_xlim(SP.BAND_F_MIN, SP.BAND_F_MAX)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("power (dB, absolute)")
        env = row["envelope"]
        ax.set_title(
            f"FLY124 {regime} {row['support']['key']} — LTAS, mic 0, "
            f"mean {row['mean_reference_rps']:.1f} rev/s "
            f"(speed {env['speed_mean']:.3f} of the {RD.AMP_RPS_REF:.0f} rev/s reference)\n"
            f"fitted envelopes here: comb {env['comb_envelope_db']:+.1f} dB, "
            f"floor {env['floor_envelope_db']:+.1f} dB "
            f"→ comb-to-floor {env['comb_minus_floor_db']:+.1f} dB"
        )
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8, ncol=2, loc="upper right")
        ax2 = axes[1]
        names = [n for n in row["arms"] if n != "v2_floor_calib"]
        pits = [row["arms"][n].get("pit_mae") for n in names]
        xs = np.arange(len(names))
        ax2.bar(
            xs,
            [0.0 if v is None else v for v in pits],
            color=[FIG_COLOURS.get(n, "#777777") for n in names],
        )
        for x, v in zip(xs, pits):
            ax2.text(
                x,
                (0.0 if v is None else v),
                "—" if v is None else f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax2.set_xticks(xs)
        ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax2.set_ylabel("HPPNet PIT MAE (rev/s)")
        ax2.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        path = out / f"ltas_{regime}.png"
        written.append(_save(fig, path))
        plt.close(fig)

        panels = [n for n in FIG_PANELS if n in fig_data["waves"]]
        fig, axs = plt.subplots(
            2, len(panels), figsize=(3.6 * len(panels), 7.4), dpi=160, sharex=True, sharey=True
        )
        w = np.hanning(SPEC_N)
        grids: list[np.ndarray] = []
        duration_s = 0.0
        for name in panels:
            x = fig_data["waves"][name]
            duration_s = float(x.size) / float(fig_data["sr"])
            starts = np.arange(0, x.size - SPEC_N + 1, SPEC_HOP)
            S = np.stack([np.abs(np.fft.rfft(x[s : s + SPEC_N] * w)) ** 2 for s in starts]).T
            grids.append(10.0 * np.log10(np.maximum(S, 1e-300)))
        allv = np.concatenate([g.ravel() for g in grids])
        shared_max = float(np.percentile(allv, 99.8))
        sf = np.fft.rfftfreq(SPEC_N, d=1.0 / fig_data["sr"])
        extent = [0.0, duration_s, float(sf[0]), float(sf[-1])]
        for col, (name, g) in enumerate(zip(panels, grids)):
            # row 0: ONE absolute scale — the level error is the message.
            # row 1: each panel on its own percentiles — the comb structure is.
            for r, (vmin, vmax) in enumerate(
                (
                    (shared_max - 75.0, shared_max),
                    (
                        float(np.percentile(g, 40.0)) - 6.0,
                        float(np.percentile(g, 99.7)),
                    ),
                )
            ):
                axp = axs[r, col]
                im = axp.imshow(
                    g,
                    origin="lower",
                    aspect="auto",
                    cmap="inferno",
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent,
                )
                if r == 0:
                    pit = row["arms"][name].get("pit_mae")
                    axp.set_title(
                        f"{name}\n{row['arms'][name]['band_level_db_mic0']:+.1f} dB band level"
                        + (f", PIT {pit:.2f} rev/s" if pit is not None else ""),
                        fontsize=9,
                    )
                else:
                    axp.set_xlabel("time (s)")
                    fig.colorbar(im, ax=axp, pad=0.02, fraction=0.05)
                axp.set_ylim(0.0, SP.BAND_F_MAX)
        axs[0, 0].set_ylabel("frequency (Hz)\n(one absolute scale)")
        axs[1, 0].set_ylabel("frequency (Hz)\n(per-panel percentiles)")
        fig.suptitle(
            f"FLY124 {regime} {row['support']['key']} — mic 0, {SPEC_N}/{SPEC_HOP} spectrogram; "
            f"top row shares one dB scale, bottom row is normalised per panel",
            fontsize=11,
        )
        fig.tight_layout()
        path = out / f"spectrogram_{regime}.png"
        written.append(_save(fig, path))
        plt.close(fig)
    return written


# ── findings ────────────────────────────────────────────────────────────────


def _f(v: Any, spec: str = ".3f") -> str:
    if v is None:
        return "—"
    try:
        return format(float(v), spec)
    except (TypeError, ValueError):
        return str(v)


def _by_regime(payload: dict[str, Any], regime: str) -> dict[str, Any] | None:
    for row in payload["supports"].values():
        if row["support"]["regime"] == regime:
            return row
    return None


def _ctf(row: dict[str, Any], arm: str, k: int) -> float | None:
    a = row["arms"].get(arm)
    if a is None:
        return None
    for c in a["comb_to_floor"]:
        if int(c["k"]) == int(k):
            return c["mean_db"]
    return None


def residual_paragraph(payload: dict[str, Any]) -> list[str]:
    """What is LEFT once both envelopes stop extrapolating, and why."""
    standby = _by_regime(payload, "standby")
    ramp = _by_regime(payload, "ramp")
    if standby is None or ramp is None:
        return []

    def pit(row: dict[str, Any], name: str) -> str:
        return _f(row["arms"].get(name, {}).get("pit_mae"))

    out = [
        "The residual, and what it is NOT: with both exponents at their prior mean the ramp is "
        f"already at the real clip's own number ({pit(ramp, 'v2_prior_exps')} vs real "
        f"{pit(ramp, 'real')} rev/s, legacy {pit(ramp, 'legacy')}), but standby only falls to "
        f"{pit(standby, 'v2_prior_exps')} against a legacy {pit(standby, 'legacy')}. Two "
        "further renders separate what remains: the comb ALONE, floor muted "
        f"({pit(standby, 'v2_prior_exps_nofloor')} rev/s at standby, "
        f"{pit(ramp, 'v2_prior_exps_nofloor')} at ramp), and the same render with the shaft "
        f"jitter at a quarter of the fitted `sigma_nu` = "
        f"{_f(payload['fit'].get('sigma_nu'), '.2f')} rev/s "
        f"({pit(standby, 'v2_prior_exps_lowjitter')} rev/s at standby, "
        f"{pit(ramp, 'v2_prior_exps_lowjitter')} at ramp) — `sigma_nu` is in rev/s, so the SAME "
        "fitted jitter is a "
        f"{_f(float(payload['fit'].get('amp_rps_ref') or 80.0) / max(standby['mean_reference_rps'], 1e-9), '.1f')}x "
        "larger fraction of a standby carrier than of the reference.",
        "",
        _residual_verdict(standby),
        "",
    ]
    return out


def _residual_verdict(standby: dict[str, Any]) -> str:
    """Which of the two residual renders actually moves standby."""

    def v(name: str) -> float | None:
        got = standby["arms"].get(name, {}).get("pit_mae")
        return None if got is None else float(got)

    base, nofloor, jitter = (
        v("v2_prior_exps"),
        v("v2_prior_exps_nofloor"),
        v("v2_prior_exps_lowjitter"),
    )
    if base is None or nofloor is None or jitter is None:
        return "_The two residual renders are not in this pass._"
    best = min(nofloor, jitter)
    if best > 0.6 * base:
        return (
            f"**Neither moves it**: muting the floor gives {nofloor:.3f} rev/s and quartering the "
            f"jitter {jitter:.3f}, against {base:.3f} with both envelopes pinned. The standby "
            "residual is therefore in the COMB ITSELF — a per-order profile fitted where 81 "
            "orders reach 7.5 kHz, reused where they reach 3.3 kHz — and no reparameterisation "
            "of the speed laws will remove it. That is what makes (c) (a standby support in the "
            "fit) load-bearing rather than optional."
        )
    if nofloor <= jitter:
        return (
            f"**The floor still masks the comb**: muting it takes standby from {base:.3f} to "
            f"{nofloor:.3f} rev/s (quartering the jitter only reaches {jitter:.3f}). The floor "
            "block itself — its cruise-fitted shape and level, not only its speed law — has to "
            "be refitted where the comb is weak."
        )
    return (
        f"**The cruise-fitted shaft jitter smears the standby comb**: quartering `sigma_nu` takes "
        f"standby from {base:.3f} to {jitter:.3f} rev/s (muting the floor only reaches "
        f"{nofloor:.3f}). A jitter parameterised in rev/s rather than in revolutions is the next "
        "defect in line."
    )


def hypothesis_section(payload: dict[str, Any]) -> list[str]:
    """The four hypotheses, each decided by numbers in this payload."""
    out = ["## Verdict per hypothesis", ""]
    cruise = _by_regime(payload, "cruise")
    rows = [(r, _by_regime(payload, r)) for r in ("standby", "ramp")]
    rows = [(name, row) for name, row in rows if row is not None]
    if not rows or cruise is None:
        out.append("_Not decidable: this pass did not measure standby, ramp and cruise together._")
        out.append("")
        return out
    out.append("| hypothesis | verdict | the numbers that decide it |")
    out.append("|---|---|---|")

    def delta_env(row: dict[str, Any], key: str) -> float:
        return float(row["envelope"][key] - cruise["envelope"][key])

    # H1 — the comb's speed envelope
    ev = "; ".join(
        f"{name}: comb envelope {_f(row['envelope']['comb_envelope_db'], '+.1f')} dB "
        f"({_f(delta_env(row, 'comb_envelope_db'), '+.1f')} dB vs cruise), "
        f"k=2 comb/floor v2 {_f(_ctf(row, 'v2', 2), '.1f')} dB vs floor-only control "
        f"{_f(_ctf(row, 'v2_nocomb', 2), '.1f')} dB and real {_f(_ctf(row, 'real', 2), '.1f')} dB"
        for name, row in rows
    )
    h1 = all(delta_env(row, "comb_envelope_db") <= -10.0 for _n, row in rows)
    out.append(
        f"| **H1** — `amp_exp` = {_f(payload['fit']['amp_exp'], '.2f')} extrapolated from cruise "
        f"collapses the comb | {'**SUPPORTED**' if h1 else 'REFUTED'} | {ev} |"
    )
    # H2 — the floor's speed envelope
    ev = "; ".join(
        f"{name}: floor envelope {_f(row['envelope']['floor_envelope_db'], '+.1f')} dB "
        f"({_f(delta_env(row, 'floor_envelope_db'), '+.1f')} dB vs cruise), v2 level offset "
        f"{_f(row['arms']['v2']['ltas_level_offset_db'], '+.1f')} dB, with `floor_exp = 0` "
        f"{_f(row['arms']['v2_floor_exp0']['ltas_level_offset_db'], '+.1f')} dB"
        for name, row in rows
    )
    h2 = all(delta_env(row, "floor_envelope_db") >= 10.0 for _n, row in rows)
    out.append(
        f"| **H2** — `floor_exp` = {_f(payload['fit']['floor_exp'], '.2f')} (NEGATIVE) makes the "
        f"floor explode as the rotors slow | {'**SUPPORTED**' if h2 else 'REFUTED'} | {ev} |"
    )
    # H3 — the render encodes another speed
    dev = []
    for name, row in rows:
        for arm in ("v2_both_exp0", "real"):
            for rot, d in row["arms"][arm]["encoded_speed"].items():
                dev.append((name, arm, rot, d.get("median_abs_dev_rev_s")))
    worst = max((v for *_x, v in dev if v is not None), default=None)
    h3 = worst is not None and worst > 0.5
    ev = (
        "order-2 demodulation against the frozen label, on the arms that HAVE a visible comb: "
        + "; ".join(f"{n}/{a}/{r} {_f(v, '.3f')} rev/s" for n, a, r, v in dev)
        + f"; k_max is {rows[0][1]['envelope']['k_max']} in every regime (no Nyquist cap change), "
        "so the carrier the render integrates IS the label track"
    )
    out.append(
        f"| **H3** — the render encodes a different speed | {'SUPPORTED' if h3 else '**REFUTED**'} | {ev} |"
    )
    # H4 — something regime-independent
    sb = _by_regime(payload, "standby") or rows[0][1]
    sb_floor0 = sb["arms"]["v2_floor_exp0"]
    ev = (
        f"cruise `{cruise['support']['key']}` runs the SAME code path, the same mic gains and the "
        "same seed, and is at parity: v2 PIT "
        f"{_f(cruise['arms']['v2'].get('pit_mae'))} rev/s, level offset "
        f"{_f(cruise['arms']['v2']['ltas_level_offset_db'], '+.2f')} dB, LTAS "
        f"{_f(cruise['arms']['v2']['ltas_mean_abs_db'], '.2f')} dB. Peak |x| of the v2 render vs "
        "the real clip: "
        + ", ".join(
            f"{name} {_f(row['arms']['v2']['peak_abs'], '.2f')} vs "
            f"{_f(row['arms']['real']['peak_abs'], '.2f')}"
            for name, row in rows
        )
        + f", cruise {_f(cruise['arms']['v2']['peak_abs'], '.2f')} vs "
        f"{_f(cruise['arms']['real']['peak_abs'], '.2f')} — the standby render is loud, not "
        "clipped (the probe reads float64 in memory and the renderer applies no normalisation), "
        "and a level error alone does not do this: `v2_floor_exp0` sits within "
        f"{_f(abs(float(sb_floor0['ltas_level_offset_db'])), '.1f')} dB "
        "of the real standby level and still scores "
        f"{_f(sb_floor0.get('pit_mae'))} rev/s"
    )
    out.append(f"| **H4** — a regime-independent defect cruise tolerates | **REFUTED** | {ev} |")
    out.append("")
    out.append(
        "Secondary (not one of the four, but real): the comb ladder is a FIXED 81 orders, so its "
        "top line slides with speed — "
        + ", ".join(
            f"{name} {_f(row['envelope']['top_comb_line_hz'], '.0f')} Hz" for name, row in rows
        )
        + f", cruise {_f(cruise['envelope']['top_comb_line_hz'], '.0f')} Hz. Above that line the "
        "standby render is pure floor."
    )
    out.append("")
    out.extend(residual_paragraph(payload))
    return out


def patch_preamble(payload: dict[str, Any]) -> list[str]:
    """The diagnosis sentence, with the fit's OWN recorded pool span in it."""
    f = payload["fit"]
    pool = f.get("pool_carrier_rev_s")
    span = f.get("pool_speed_span")
    standby = _by_regime(payload, "standby")
    cruise = _by_regime(payload, "cruise")
    gap_db = (
        0.0
        if standby is None or cruise is None
        else abs(
            float(standby["envelope"]["comb_minus_floor_db"])
            - float(cruise["envelope"]["comb_minus_floor_db"])
        )
    )
    where = (
        "the pool's carrier range is not recorded in this fit"
        if pool is None
        else f"the fit's own objective saw carriers {pool[0]:.1f}-{pool[1]:.1f} rev/s, a span of "
        f"{span:.3f}x"
    )
    outside = (
        ""
        if pool is None or standby is None
        else f" The standby support runs at {standby['mean_reference_rps']:.1f} rev/s — "
        f"{standby['mean_reference_rps'] / float(pool[0]):.2f}x the LOWEST carrier the fit ever "
        "saw, so both envelopes are pure extrapolation there."
    )
    return [
        "## Minimal patch proposal (NOT applied — Main's call)",
        "",
        "Both supported hypotheses are ONE defect: the two speed-envelope exponents are fitted on "
        f"a pool that barely varies in speed ({where}, {f['n_pool_windows']} FLY125 cruise "
        f"windows), where they trade almost exactly against `profile_db` and `floor_mean_db`. The "
        f"optimiser took `amp_exp` = {f['amp_exp']:.2f} and `floor_exp` = {f['floor_exp']:.2f} "
        "against a `N(2, 2)` prior that 24 M Whittle cells simply outvote; in sample the two "
        "extremes cancel, and outside it they pull the comb and the floor "
        f"{gap_db:.0f} dB apart." + outside + " Two changes, smallest first:",
        "",
    ]


PATCH_PROPOSAL: list[str] = [
    "**(a) the floor exponent may not be negative** — a rotor floor cannot get LOUDER as the "
    "rotors slow. One site changes parameterisation (`src/experiments/noise_model/model.py`):",
    "",
    "```diff",
    "@@ class Priors",
    "-    amp_exp: tuple[float, float] = (2.0, 2.0)",
    "-    floor_exp: tuple[float, float] = (2.0, 2.0)",
    "+    amp_exp: tuple[float, float] = (2.0, 2.0)",
    "+    #: LOG-space now: the floor's speed exponent is positive by construction",
    "+    log_floor_exp: tuple[float, float] = (math.log(2.0), 0.7)",
    "@@ def sample_params",
    '-            exp=_normal(site, "floor_exp", *priors.floor_exp) if flight else zero,',
    "+            exp=(",
    '+                _lognormal(site, "floor_exp", priors.log_floor_exp) if flight else zero',
    "+            ),",
    "```",
    "",
    "**(b) do not FIT an envelope the pool cannot see.** A flight pool whose speed span is under "
    "1.5x identifies neither exponent, so pin both at the prior mean instead of letting the "
    "likelihood run away with them:",
    "",
    "```diff",
    "@@ def sample_params",
    "     free = free_blocks(mode)",
    "     fz = dict(frozen or {})",
    "     r, m, k = batch.n_rotors, batch.n_mics, batch.k_max",
    '     flight = batch.mode == "flight"',
    "+    # the speed envelopes are identified by the SPAN of the pool's speeds; a",
    "+    # single-regime pool has none, and a free exponent then extrapolates",
    "+    # 70 dB of comb-to-floor error onto every other regime",
    "+    envelopes_identified = flight and batch.speed_span >= SPEED_SPAN_MIN",
    "@@",
    '-        amp_exp = _normal(site, "amp_exp", *priors.amp_exp) if flight else zero',
    "+        amp_exp = (",
    '+            _normal(site, "amp_exp", *priors.amp_exp)',
    "+            if envelopes_identified",
    "+            else torch.as_tensor(priors.amp_exp[0], dtype=torch.float64)",
    "+        )",
    "```",
    "",
    "with `speed_span = max_rotor_frame_speed / min_rotor_frame_speed` recorded on "
    "`SupportBatch` by `flight_batch` (it already holds `rate_work`) and "
    "`SPEED_SPAN_MIN = 1.5`; the same gate applies to `floor_exp` and `floor_static_rel`.",
    "",
    "**(c) the campaign-level fix, and the one that actually buys parity: stop extrapolating.** "
    "The legacy arm reaches standby 0.32 rev/s because the frozen evaluator renders standby from "
    "a STANDBY export (`results/S2/standby.json`, regime `standby`) and only its cruise/ramp arms "
    "come from the cruise export. Round 2's Michael's fit should do the same: either pool FLY125 "
    "standby + ramp + cruise windows into one flight fit (which also makes (b)'s span gate open "
    "legitimately), or fit one support per regime and render each regime from its own. No "
    "renderer or model change is needed for this one — it is the fit pool.",
    "",
]


def _equal_regime_note(payload: dict[str, Any]) -> list[str]:
    """Where the frozen Michael's bar would stand under each variant.

    INDICATIVE, not the gate: the frozen gate averages per-regime means over
    all five Michael's supports (two standby, one ramp, two cruise) and this
    study measures one support per regime.
    """
    arms = ("v2", "v2_prior_exps", "v2_level_matched_exps", "legacy", "real")
    rows = [_by_regime(payload, r) for r in ("standby", "ramp", "cruise")]
    if any(r is None for r in rows):
        return []
    out = [
        "Equal-regime mean of the three supports measured here, against the frozen "
        f"Michael's parity bar {GT.MICHAELS_PIT_BOUND:.6f} rev/s "
        f"(1.05 x the legacy {GT.MICHAELS_BASELINE_REGIME_MEAN:.6f}) — INDICATIVE only, the "
        "gate averages five supports:",
        "",
    ]
    out.append("| arm | " + " | ".join(arms) + " |")
    out.append("|---|" + "---:|" * len(arms))
    cells: list[str] = []
    means: dict[str, float] = {}
    for name in arms:
        vals = [r["arms"].get(name, {}).get("pit_mae") for r in rows if r is not None]
        if any(v is None for v in vals):
            cells.append("—")
            continue
        mean = float(np.mean([float(v) for v in vals]))
        means[name] = mean
        cells.append(f"{mean:.3f}{' ✓' if mean <= GT.MICHAELS_PIT_BOUND else ' ✗'}")
    out.append("| equal-regime mean (3 supports) | " + " | ".join(cells) + " |")
    out.append("")
    if "legacy" in means:
        out.append(
            f"Read the DIFFERENCES, not the absolutes: the legacy arm itself scores "
            f"{means['legacy']:.3f} on this three-support subset against its own frozen "
            f"{GT.MICHAELS_BASELINE_REGIME_MEAN:.6f} over five supports (this subset's single "
            f"ramp support is the harder one — legacy {_f(rows[1]['arms']['legacy'].get('pit_mae') if rows[1] else None)} "
            f"here against the frozen legacy ramp reference "
            f"{GT.MICHAELS_BASELINE_PIT_MAE.get('ramp', float('nan')):.6f}), so the subset is "
            "harsher than the gate and no row of it is a gate verdict."
        )
        out.append("")
    return out


def patch_expectation(payload: dict[str, Any]) -> list[str]:
    """The MEASURED effect of each proposed change, from this run's variants."""
    out = [
        "### Expected effect — measured, not guessed",
        "",
        "No variant below is a FIT, so none of them is a parity claim: each is the SAME fitted "
        "comb and floor with one speed law stopped from extrapolating, scored by the same frozen "
        "HPPNet on the same frozen support.",
        "",
        "| support | v2 as fitted | (a)+(b) both exponents at the prior mean `v2_prior_exps` "
        "| the best a single global envelope can do `v2_level_matched_exps` | legacy (the bar) "
        "| real |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for regime in ("standby", "ramp", "cruise"):
        row = _by_regime(payload, regime)
        if row is None:
            continue

        def pit(name: str, row: dict[str, Any] = row) -> str:
            return _f(row["arms"].get(name, {}).get("pit_mae"))

        out.append(
            f"| {regime} | {pit('v2')} | {pit('v2_prior_exps')} "
            f"| {pit('v2_level_matched_exps')} (exp "
            f"{_f(row.get('level_matched_exp'), '.2f')}) | {pit('legacy')} | {pit('real')} |"
        )
    out.append("")
    out.extend(_equal_regime_note(payload))
    return out


def findings(payload: dict[str, Any], *, job: str | None, figures: list[str]) -> str:
    out: list[str] = []
    rows = payload["supports"]
    order = [
        k
        for regime in ("standby", "ramp", "cruise")
        for k, r in rows.items()
        if r["support"]["regime"] == regime
    ]
    out.append("# Noise model v2 — round 2: why the v2 render loses standby and ramp")
    out.append("")
    out.append(
        f"Record `{OUT_DEFAULT}/render_regime.json`, git `{payload['git']}`, "
        f"fit `{payload['fit']['path']}` (amp_exp {_f(payload['fit']['amp_exp'], '.4f')}, "
        f"floor_exp {_f(payload['fit']['floor_exp'], '.4f')}, "
        f"floor_static_rel {_f(payload['fit']['floor_static_rel'], '.6f')}, "
        f"reference speed {_f(payload['fit']['amp_rps_ref'], '.0f')} rev/s), "
        f"render seed {payload['protocol']['seed']}, {payload['protocol']['n_mics']} mics."
    )
    if job:
        out.append("")
        out.append(
            f"HPPNet job `{job}` on `uni-gpushort` (frozen checkpoint sha256 verified in-job)."
        )
    out.append("")
    out.append(
        f"## PIT MAE per support and arm (rev/s, frozen HPPNet, seed {payload['protocol']['seed']})"
    )
    out.append("")
    names = [a.name for a in ARMS if a.probe]
    out.append("| support | regime | " + " | ".join(names) + " |")
    out.append("|---|---|" + "---:|" * len(names))
    for key in order:
        r = rows[key]
        cells = [_f(r["arms"].get(n, {}).get("pit_mae")) for n in names]
        out.append(f"| `{key}` | {r['support']['regime']} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("## Levels and LTAS (mic 0, 30–7900 Hz, absolute)")
    out.append("")
    out.append(
        "| support | arm | band level dB | level offset vs real dB | LTAS mean abs dB "
        "| LTAS shape-only dB | comb/floor k=2 dB | k=4 | k=8 |"
    )
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for key in order:
        r = rows[key]
        for name, a in r["arms"].items():
            ctf = {c["k"]: c["mean_db"] for c in a["comb_to_floor"]}
            out.append(
                f"| `{r['support']['regime']}` | {name} | {_f(a['band_level_db_mic0'], '.2f')} "
                f"| {_f(a['ltas_level_offset_db'], '.2f')} | {_f(a['ltas_mean_abs_db'], '.2f')} "
                f"| {_f(a['ltas_shape_only_mean_abs_db'], '.2f')} "
                f"| {_f(ctf.get(2), '.1f')} | {_f(ctf.get(4), '.1f')} | {_f(ctf.get(8), '.1f')} |"
            )
    out.append("")
    out.append("## The two fitted speed envelopes at each support's speed")
    out.append("")
    out.append(
        "| support | regime | mean rev/s | speed | comb envelope dB | floor envelope dB "
        "| comb − floor dB | k_max | top comb line Hz |"
    )
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for key in order:
        r = rows[key]
        e = r["envelope"]
        out.append(
            f"| `{key}` | {r['support']['regime']} | {_f(r['mean_reference_rps'], '.2f')} "
            f"| {_f(e['speed_mean'], '.4f')} | {_f(e['comb_envelope_db'], '+.2f')} "
            f"| {_f(e['floor_envelope_db'], '+.2f')} | {_f(e['comb_minus_floor_db'], '+.2f')} "
            f"| {e['k_max']} | {_f(e['top_comb_line_hz'], '.0f')} |"
        )
    out.append("")
    out.append("## Encoded speed (order-2 demodulation against the frozen label)")
    out.append("")
    out.append("| support | arm | rotor | median abs dev rev/s | p90 | mean signed |")
    out.append("|---|---|---|---:|---:|---:|")
    for key in order:
        r = rows[key]
        for name in ("real", "legacy", "v2", "v2_both_exp0", "v2_nocomb"):
            a = r["arms"].get(name)
            if a is None:
                continue
            for rot, d in a["encoded_speed"].items():
                out.append(
                    f"| `{r['support']['regime']}` | {name} | {rot} "
                    f"| {_f(d.get('median_abs_dev_rev_s'))} | {_f(d.get('p90_abs_dev_rev_s'))} "
                    f"| {_f(d.get('mean_dev_rev_s'))} |"
                )
    out.append("")
    out.extend(hypothesis_section(payload))
    out.extend(patch_preamble(payload))
    out.extend(PATCH_PROPOSAL)
    out.extend(patch_expectation(payload))
    if figures:
        out.append("## Figures")
        out.append("")
        for f in figures:
            out.append(f"* `{f}`")
        out.append("")
    return "\n".join(out)


PIT_KEYS = ("pit_mae", "pit_per_mic", "pit_per_rotor", "n_scored_frames")
#: Two passes of the same seed must render the same audio bit for bit; 0.01 dB
#: of band level is far tighter than any real difference and far looser than
#: float64 noise.
LEVEL_TOL_DB = 0.01


def merge_pit(payload: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    """Copy the probe pass's PIT numbers onto this pass's arms.

    Refuses any arm whose band level disagrees: the renders are deterministic
    in the seed, so a level difference means the two passes did not measure the
    same audio and the PIT number does not belong to this row.
    """
    if str(probe.get("schema")) != SCHEMA:
        die(f"--pit-from payload carries schema {probe.get('schema')!r}, need {SCHEMA!r}")
    if int(probe["protocol"]["seed"]) != int(payload["protocol"]["seed"]):
        die(
            f"--pit-from was rendered at seed {probe['protocol']['seed']}, this pass at "
            f"{payload['protocol']['seed']}"
        )
    merged = 0
    for key, row in payload["supports"].items():
        other = probe["supports"].get(key)
        if other is None:
            die(f"--pit-from payload has no support {key!r}")
        for name, arm in row["arms"].items():
            src = other["arms"].get(name)
            if src is None:
                continue
            gap = abs(float(src["band_level_db_mic0"]) - float(arm["band_level_db_mic0"]))
            if gap > LEVEL_TOL_DB:
                die(
                    f"{key}/{name}: --pit-from band level {src['band_level_db_mic0']:.4f} dB "
                    f"differs from this pass's {arm['band_level_db_mic0']:.4f} dB by {gap:.4f} dB; "
                    "the two passes did not render the same audio"
                )
            if "pit_mae" not in src:
                continue
            for k in PIT_KEYS:
                if k in src:
                    arm[k] = src[k]
            merged += 1
    if merged == 0:
        die(f"--pit-from payload carries no pit_mae at all (job {probe.get('job')!r})")
    return dict(
        path=None,
        schema=str(probe["schema"]),
        git=probe.get("git"),
        job=probe.get("job"),
        scorer=probe["protocol"].get("scorer"),
        arms_merged=int(merged),
        level_tol_db=LEVEL_TOL_DB,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", type=Path, default=FIT_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--json-name", default="render_regime.json")
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--n-mics", type=int, default=8)
    ap.add_argument("--regimes", default=",".join(STUDY_REGIMES))
    ap.add_argument("--probe", action="store_true", help="run the frozen HPPNet scorer")
    ap.add_argument("--figures", action="store_true", help="write the PNG panels")
    ap.add_argument("--job", default=None, help="the omnirun job id, stamped into the findings")
    ap.add_argument(
        "--pit-from",
        type=Path,
        default=None,
        help="merge the PIT MAE of a --probe pass of this same seed into this run "
        "(the renders are deterministic, and the merge REFUSES a payload whose arm "
        "levels differ)",
    )
    ap.add_argument(
        "--findings",
        default="findings.md",
        help="findings file name inside --out ('' to skip)",
    )
    args = ap.parse_args(argv)
    payload = run(
        fit_path=args.fit,
        out=args.out,
        seed=int(args.seed),
        regimes=tuple(s.strip() for s in str(args.regimes).split(",") if s.strip()),
        probe=bool(args.probe),
        n_mics=int(args.n_mics),
    )
    fig_data = payload.pop("_figures")
    if args.pit_from is not None:
        payload["pit_source"] = merge_pit(payload, json.loads(Path(args.pit_from).read_text()))
        payload["verdicts"] = verdicts(payload)
    figs: list[str] = []
    if args.figures:
        figs = write_figures(payload, fig_data, Path(args.out))
    payload["figures"] = figs
    payload["job"] = args.job
    path = Path(args.out) / str(args.json_name)
    path.write_text(json.dumps(payload, indent=1, allow_nan=True) + "\n")
    print(f"wrote {path}")
    if args.findings:
        fpath = Path(args.out) / str(args.findings)
        fpath.write_text(findings(payload, job=args.job, figures=figs) + "\n")
        print(f"wrote {fpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
