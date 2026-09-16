"""Noise-model-v2 criteria study B: which spectrogram-similarity proxy can
carry a threshold, and where its oracle floor sits.

FIVE candidate distances between a synthetic clip and a real clip on the SAME
support, all on mic 0 and all inside the frozen 30-7900 Hz band:

``a`` ``mr_ltas``
    multi-resolution LTAS: the L1 over bins of the time-averaged log
    magnitude, at NFFT 512 / 2048 / 8192, UNWHITENED and with no per-arm
    normalisation. Units dB per bin.
``b`` ``mr_ltas_whitened``
    the same, after both clips are divided by ONE whitener -- the REAL clip's
    own LTAS at NFFT 8192, smoothed over 1/6 octave -- and floored. Equal band
    weighting, no per-arm normalisation. The script also records the exact
    identity ``b == a`` in the UNFLOORED limit, because dividing both clips by
    the same per-bin constant cancels in a difference of logs: the floor is
    the only thing that makes ``b`` a different measurement, and that is
    reported as a number rather than assumed.
``c`` ``texture``
    the per-bin temporal standard deviation of the whitened log magnitude at
    NFFT 2048, L1 over bins. A clip whose lines wander at the wrong rate has
    the wrong temporal spread even when its LTAS is right.
``d`` ``modulation``
    per-bin modulation spectrum of the whitened log-magnitude envelope
    (0.5-50 Hz) at NFFT 2048, each band normalised to unit modulation power,
    L1 over bands. Dimensionless in ``[0, 2]``.
``e`` ``ltas_abs_db``
    the FROZEN absolute-level band LTAS error of
    ``revised_eval.ltas_deviation_db`` -- the reference the campaign already
    gates on, carried here unchanged so every new candidate is comparable to
    something already accepted.

ARMS, every one of them scored against the same real clip:

* ``oracle`` -- the real clip against a DISJOINT real segment of the same
  recording, truncated to the same length. The floor: no model can do better
  than one piece of real material predicting another;
* ``current_best`` -- the legacy stage-2 export the frozen evaluator selects,
  rendered on the real carrier at physical level;
* ``c3`` -- the round-3 C3 revised export, rendered on the real carrier;
* a controlled DEGRADATION LADDER -- the C3 export re-rendered with
  ``d_scalar`` and with ``sigma`` scaled. The ladder base is the revised
  export because the legacy current-best arm has no ``D`` and no ``sigma`` to
  scale: its phase model is a fixed coherent/Lorentzian mixture. The ladder
  therefore measures how each candidate responds to the two parameters model
  v2 has to identify, which is exactly what a proxy must be sensitive to.

For every candidate the script reports the oracle floor and its spread across
supports, the current-best and C3 values, the ladder's Spearman rho against
the scale, and the separation ratio
``(current_best - oracle) / (oracle spread across supports)``.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy import stats as sstats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from noise_v2_likelihood_window import (  # noqa: E402
    BAND_HZ,
    C3_EXPORTS,
    DATASET,
    DREGON_SCORED,
    GUARD_S,
    MICHAELS_SCORED,
    RAW_KEY,
    SR,
    Support,
    baseline_params,
    disjoint_segment,
    git_head,
    load,
)

from experiments.stochastic_fit import clips as C  # noqa: E402
from experiments.stochastic_fit import revised_eval as RE  # noqa: E402
from experiments.stochastic_fit import revised_phase as RP  # noqa: E402
from experiments.stochastic_fit import stage2 as S2  # noqa: E402
from experiments.stochastic_fit.data import Clip  # noqa: E402

TOPIC = "criteria"
OUT_DEFAULT = Path("results/noise_v2") / TOPIC
FIG_DEFAULT = Path("docs/explainers/noise-model-v2-plan")

#: The analysed microphone. One mic, stated, because the frozen LTAS gate
#: (``revised_eval.ltas_deviation_db``) reads mic 0 and candidate ``e`` must
#: be comparable to it.
MIC = 0

#: The multi-resolution LTAS geometries of candidates (a) and (b).
LTAS_NFFTS = (512, 2048, 8192)
#: The whitener's own geometry: computed ONCE per support from the REAL clip.
WHITENER_NFFT = 8192
WHITENER_OCTAVE = 1.0 / 6.0
#: Texture and modulation geometry.
TEXTURE_NFFT = 2048
#: Analysis hop of every spectrogram here: NFFT/16, the frozen overlap ratio.
HOP_RATIO = 16
#: Floor of the whitened log magnitude, in dB below the real clip's own
#: smoothed level. Without a floor the whitener cancels exactly in every
#: log-domain difference; with it, bins the real clip does not excite stop
#: contributing unbounded negative dB.
WHITENED_FLOOR_DB = -40.0
#: Modulation band of candidate (d).
MOD_BAND_HZ = (0.5, 50.0)

#: The render seed. One of the frozen ``null_variation.seeds`` of
#: docs/revised-phase-baseline-manifest-v2.json, so no new seed convention is
#: invented here.
RENDER_SEED = 2001

D_SCALES = (1.0 / 30.0, 1.0 / 10.0, 1.0 / 3.0, 1.0, 3.0, 10.0, 30.0)
SIGMA_SCALES = (1.0 / 10.0, 1.0 / 3.0, 1.0, 3.0, 10.0)

#: Michael's second real segment: a FLY125 cruise window. It enters ONLY the
#: real-vs-real oracle pair (no model touches it), so using fit material here
#: leaks nothing into any scored arm.
MICHAELS_SECOND_REAL = ("FLY125", 32.0)


# ── spectrograms and the whitener ───────────────────────────────────────────


def spectrogram(x: np.ndarray, n_fft: int, *, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    """``(freqs, |X|^2)`` with the project's periodic Hann and an NFFT/16 hop."""
    xx = np.asarray(x, dtype=np.float64)
    hop = max(n_fft // HOP_RATIO, 1)
    window = np.hanning(n_fft + 1)[:n_fft]
    if xx.size < n_fft:
        raise ValueError(f"{xx.size} samples is shorter than the {n_fft}-point window")
    starts = np.arange(1 + (xx.size - n_fft) // hop) * hop
    frames = np.stack([xx[s : s + n_fft] for s in starts]) * window
    spec = np.fft.rfft(frames, axis=-1)
    power = (spec.real**2 + spec.imag**2) / float(np.sum(window**2))
    return np.fft.rfftfreq(n_fft, 1.0 / sr), power  # (N, F)


def band_mask(freqs: np.ndarray) -> np.ndarray:
    f = np.asarray(freqs, dtype=np.float64)
    return (f >= BAND_HZ[0]) & (f <= BAND_HZ[1])


def smooth_octave(freqs: np.ndarray, power: np.ndarray, octaves: float) -> np.ndarray:
    """Fractional-octave smoothing of a power spectrum, in the power domain."""
    f = np.asarray(freqs, dtype=np.float64)
    p = np.asarray(power, dtype=np.float64)
    out = np.empty_like(p)
    ratio = 2.0 ** (0.5 * float(octaves))
    # bin 0 has no octave neighbourhood; hold the first positive-frequency value
    for i, fc in enumerate(f):
        if fc <= 0.0:
            out[i] = p[1] if p.size > 1 else p[i]
            continue
        sel = (f >= fc / ratio) & (f <= fc * ratio)
        out[i] = p[sel].mean() if sel.any() else p[i]
    return out


def whitener(real: np.ndarray, *, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    """``(freqs, W)``: the REAL clip's LTAS at NFFT 8192, 1/6-octave smoothed.

    Computed ONCE per support and applied identically to both clips of every
    pair, so it can never act as a per-arm normalisation.
    """
    freqs, power = spectrogram(real, WHITENER_NFFT, sr=sr)
    return freqs, smooth_octave(freqs, power.mean(axis=0), WHITENER_OCTAVE)


def whitener_on(freqs_w: np.ndarray, w: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    return np.interp(np.asarray(freqs), np.asarray(freqs_w), np.asarray(w))


def log_mag(power: np.ndarray, *, w: np.ndarray | None, floor_db: float | None) -> np.ndarray:
    """``10 log10`` of a (whitened, floored) power spectrogram."""
    p = np.asarray(power, dtype=np.float64)
    if w is not None:
        p = p / np.maximum(w[None, :], 1e-300)
    db = 10.0 * np.log10(np.maximum(p, 1e-300))
    if floor_db is not None:
        db = np.maximum(db, float(floor_db))
    return db


# ── the five candidates ─────────────────────────────────────────────────────


def cand_mr_ltas(
    real: np.ndarray,
    arm: np.ndarray,
    *,
    w_freqs: np.ndarray | None,
    w: np.ndarray | None,
    floor_db: float | None,
) -> dict[str, Any]:
    """L1 of the time-averaged log magnitude per bin, at each NFFT."""
    per: dict[str, float] = {}
    for n_fft in LTAS_NFFTS:
        freqs, pr = spectrogram(real, n_fft)
        _, pa = spectrogram(arm, n_fft)
        n = min(pr.shape[0], pa.shape[0])
        wg = None if w is None else whitener_on(w_freqs, w, freqs)  # type: ignore[arg-type]
        lr = log_mag(pr[:n], w=wg, floor_db=floor_db).mean(axis=0)
        la = log_mag(pa[:n], w=wg, floor_db=floor_db).mean(axis=0)
        m = band_mask(freqs)
        per[str(n_fft)] = float(np.abs(lr[m] - la[m]).mean())
    return dict(value=float(np.mean(list(per.values()))), per_nfft=per)


def line_bin_mask(freqs: np.ndarray, rates: np.ndarray, *, half_bins: float = 0.5) -> np.ndarray:
    """Bins within ``half_bins`` of ANY telemetry comb line ``k * f_r``.

    The phase model only acts on the comb; a texture or modulation statistic
    averaged over the whole band is mostly floor. This mask is the LINE
    subset, built from the support's own mean telemetry rates and nothing
    fitted.
    """
    f = np.asarray(freqs, dtype=np.float64)
    df = float(f[1] - f[0])
    out = np.zeros(f.shape, dtype=bool)
    for rate in np.atleast_1d(np.asarray(rates, dtype=np.float64)):
        if rate <= 0.0:
            continue
        k_max = int(np.floor(BAND_HZ[1] / rate))
        for k in range(1, max(k_max, 1) + 1):
            out |= np.abs(f - k * rate) <= half_bins * df
    return out


def cand_texture(
    real: np.ndarray,
    arm: np.ndarray,
    *,
    w_freqs: np.ndarray,
    w: np.ndarray,
    rates: np.ndarray | None = None,
) -> dict[str, Any]:
    """L1 of the per-bin temporal standard deviation of the whitened log magnitude."""
    freqs, pr = spectrogram(real, TEXTURE_NFFT)
    _, pa = spectrogram(arm, TEXTURE_NFFT)
    n = min(pr.shape[0], pa.shape[0])
    wg = whitener_on(w_freqs, w, freqs)
    sr_ = log_mag(pr[:n], w=wg, floor_db=WHITENED_FLOOR_DB).std(axis=0)
    sa = log_mag(pa[:n], w=wg, floor_db=WHITENED_FLOOR_DB).std(axis=0)
    m = band_mask(freqs)
    if rates is not None:
        m = m & line_bin_mask(freqs, rates)
    return dict(
        value=float(np.abs(sr_[m] - sa[m]).mean()),
        real_mean_std_db=float(sr_[m].mean()),
        arm_mean_std_db=float(sa[m].mean()),
        n_bins=int(m.sum()),
        n_frames=int(n),
    )


def modulation_shape(
    db: np.ndarray, *, frame_rate: float, band: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-band normalised modulation PSD of a log-magnitude spectrogram."""
    from scipy import signal as _sig

    x = np.asarray(db, dtype=np.float64)[:, band]
    x = x - x.mean(axis=0, keepdims=True)
    nper = int(min(256, x.shape[0] // 2))
    if nper < 16:
        return None
    win = np.hanning(nper + 1)[:nper]
    # the scipy stubs declare `window: str` and `detrend: str`; the array window
    # and `detrend=False` are the documented runtime forms, passed as kwargs
    kw: dict[str, Any] = dict(
        window=win,
        nperseg=nper,
        noverlap=nper // 2,
        detrend=False,
        scaling="density",
    )
    freqs, psd = _sig.welch(x, fs=float(frame_rate), axis=0, **kw)
    keep = (freqs >= MOD_BAND_HZ[0]) & (freqs <= MOD_BAND_HZ[1])
    p = np.asarray(psd)[keep]
    total = p.sum(axis=0, keepdims=True)
    return np.asarray(freqs)[keep], p / np.maximum(total, 1e-300)


def cand_modulation(
    real: np.ndarray,
    arm: np.ndarray,
    *,
    w_freqs: np.ndarray,
    w: np.ndarray,
    rates: np.ndarray | None = None,
) -> dict[str, Any]:
    """L1 between per-band normalised modulation spectra (0.5-50 Hz)."""
    freqs, pr = spectrogram(real, TEXTURE_NFFT)
    _, pa = spectrogram(arm, TEXTURE_NFFT)
    n = min(pr.shape[0], pa.shape[0])
    wg = whitener_on(w_freqs, w, freqs)
    m = band_mask(freqs)
    if rates is not None:
        m = m & line_bin_mask(freqs, rates)
    frame_rate = SR / float(max(TEXTURE_NFFT // HOP_RATIO, 1))
    a = modulation_shape(
        log_mag(pr[:n], w=wg, floor_db=WHITENED_FLOOR_DB), frame_rate=frame_rate, band=m
    )
    b = modulation_shape(
        log_mag(pa[:n], w=wg, floor_db=WHITENED_FLOOR_DB), frame_rate=frame_rate, band=m
    )
    if a is None or b is None:
        return dict(value=None, reason="fewer than 32 analysis frames: no modulation estimate")
    mf, pa_n = a
    _, pb_n = b
    return dict(
        value=float(np.abs(pa_n - pb_n).sum(axis=0).mean()),
        modulation_df_hz=float(mf[1] - mf[0]) if mf.size > 1 else None,
        n_modulation_bins=int(mf.size),
        n_bins=int(m.sum()),
        n_frames=int(n),
    )


def candidates(
    real: np.ndarray,
    arm: np.ndarray,
    *,
    w_freqs: np.ndarray,
    w: np.ndarray,
    rates: np.ndarray,
    full: bool = True,
) -> dict[str, Any]:
    """Every candidate distance of one (real, arm) pair on one support."""
    out: dict[str, Any] = {}
    out["mr_ltas"] = cand_mr_ltas(real, arm, w_freqs=None, w=None, floor_db=None)
    out["mr_ltas_whitened"] = cand_mr_ltas(
        real, arm, w_freqs=w_freqs, w=w, floor_db=WHITENED_FLOOR_DB
    )
    if full:
        unfloored = cand_mr_ltas(real, arm, w_freqs=w_freqs, w=w, floor_db=None)
        out["mr_ltas_whitened"]["unfloored_value"] = unfloored["value"]
        out["mr_ltas_whitened"]["unfloored_minus_unwhitened"] = float(
            unfloored["value"] - out["mr_ltas"]["value"]
        )
    out["texture"] = cand_texture(real, arm, w_freqs=w_freqs, w=w)
    out["texture_lines"] = cand_texture(real, arm, w_freqs=w_freqs, w=w, rates=rates)
    out["modulation"] = cand_modulation(real, arm, w_freqs=w_freqs, w=w)
    out["modulation_lines"] = cand_modulation(real, arm, w_freqs=w_freqs, w=w, rates=rates)
    two = np.stack([real, real]) if real.ndim == 1 else real
    arm_two = np.stack([arm, arm]) if arm.ndim == 1 else arm
    dev = RE.ltas_deviation_db(two, arm_two, mic=0)
    out["ltas_abs_db"] = dict(
        value=float(dev["mean_abs_db"]),
        shape_only_mean_abs_db=float(dev["shape_only_mean_abs_db"]),
        level_offset_db=float(dev["level_offset_db"]),
        max_abs_db=float(dev["max_abs_db"]),
    )
    return out


#: The five candidates the study was asked for, plus the two LINE-restricted
#: variants of (c) and (d) -- the comb is where the phase model acts, and a
#: whole-band average of a texture statistic is mostly broadband floor.
CANDIDATE_NAMES = (
    "mr_ltas",
    "mr_ltas_whitened",
    "texture",
    "modulation",
    "ltas_abs_db",
    "texture_lines",
    "modulation_lines",
)
CANDIDATE_UNITS = {
    "mr_ltas": "dB per bin",
    "mr_ltas_whitened": "dB per bin (whitened, floored)",
    "texture": "dB per bin (std of whitened log magnitude)",
    "modulation": "L1 of normalised modulation shape, dimensionless",
    "ltas_abs_db": "dB per frozen band",
    "texture_lines": "dB per comb bin (std of whitened log magnitude, comb bins only)",
    "modulation_lines": "L1 of normalised modulation shape, comb bins only",
}


# ── arms ────────────────────────────────────────────────────────────────────


@dataclass
class Arm:
    name: str
    kind: str
    audio: np.ndarray
    detail: dict[str, Any]
    #: when set, THIS is the real side of the pair instead of the whole
    #: scored clip (the split-half oracle needs both sides to be halves)
    real_override: np.ndarray | None = None


def scaled_export(
    summary: dict[str, Any], *, d_scale: float = 1.0, sigma_scale: float = 1.0
) -> dict[str, Any]:
    out = copy.deepcopy(summary)
    out["parameters"]["d_scalar"] = float(out["parameters"]["d_scalar"]) * float(d_scale)
    out["parameters"]["sigma"] = float(out["parameters"]["sigma"]) * float(sigma_scale)
    return out


def build_arms(
    *,
    rig: str,
    scored: Support,
    real_clip: Clip,
    ref_clip: Clip | None,
    second_real: np.ndarray | None,
    c3_summary: dict[str, Any],
    ladder: bool,
    n_mics: int,
) -> list[Arm]:
    rps = np.asarray(real_clip.rps, dtype=np.float64)
    n = int(real_clip.audio.shape[1])
    arms: list[Arm] = []
    if ref_clip is not None:
        m = min(n, int(ref_clip.audio.shape[1]))
        arms.append(
            Arm(
                "oracle",
                "real",
                np.asarray(ref_clip.audio, dtype=np.float64)[:, :m],
                dict(
                    reference=ref_clip.clip_id,
                    n_samples=m,
                    note="speed-matched disjoint real segment of the same recording, truncated "
                    "to the common length",
                ),
            )
        )
    half = n // 2
    if half >= 2 * WHITENER_NFFT:
        arms.append(
            Arm(
                "oracle_split_half",
                "real",
                np.asarray(real_clip.audio, dtype=np.float64)[:, half : 2 * half],
                dict(
                    n_samples=half,
                    note="the scored window's OWN two halves against each other: the tightest "
                    "real-vs-real floor, at half the length of every other arm",
                ),
                real_override=np.asarray(real_clip.audio, dtype=np.float64)[:, :half],
            )
        )
    if second_real is not None:
        m = min(n, int(second_real.shape[1]))
        arms.append(
            Arm(
                "oracle_cross_recording",
                "real",
                np.asarray(second_real, dtype=np.float64)[:, :m],
                dict(
                    n_samples=m,
                    note="second REAL segment from the other recording of the same rig; "
                    "real-vs-real only, no model touches it",
                ),
            )
        )
    mp = baseline_params(rig, scored.regime, scored.recording)
    physical = RE.to_renderer_units(mp)
    arms.append(
        Arm(
            "current_best",
            "legacy",
            np.asarray(
                S2.render_from_export(
                    dict(physical.params),
                    rps,
                    sample_rate_work=RE.SAMPLE_RATE_WORK,
                    n_mics=n_mics,
                    seed=RENDER_SEED,
                    normalize_rms=None,
                ),
                dtype=np.float64,
            ),
            dict(
                source=dict(mp.source),
                observation_law=RE.HISTORICAL_FORWARD_LABEL,
                seed=RENDER_SEED,
            ),
        )
    )
    specs: list[tuple[str, float, float]] = [("c3", 1.0, 1.0)]
    if ladder:
        specs += [(f"d_x{s:g}", s, 1.0) for s in D_SCALES if s != 1.0]
        specs += [(f"sigma_x{s:g}", 1.0, s) for s in SIGMA_SCALES if s != 1.0]
    for name, ds, ss in specs:
        summary = (
            c3_summary
            if (ds == 1.0 and ss == 1.0)
            else scaled_export(c3_summary, d_scale=ds, sigma_scale=ss)
        )
        out = RP.render_revised(
            summary,
            rps,
            sample_rate=SR,
            sample_rate_work=RE.SAMPLE_RATE_WORK,
            n_mics=n_mics,
            seed=RENDER_SEED,
        )
        arms.append(
            Arm(
                name,
                "revised",
                np.asarray(out.audio, dtype=np.float64),
                dict(
                    d_scale=float(ds),
                    sigma_scale=float(ss),
                    d_scalar=float(summary["parameters"]["d_scalar"]),
                    sigma=float(summary["parameters"]["sigma"]),
                    seed=RENDER_SEED,
                ),
            )
        )
    return arms


# ── the grid ────────────────────────────────────────────────────────────────


def run(
    *, rigs: list[str], supports_per_rig: int | None, ladder: bool, n_mics: int
) -> dict[str, Any]:
    payload: dict[str, Any] = dict(
        schema="noise-v2-criteria-proxy/1",
        front_end=dict(
            mic=MIC,
            band_hz=list(BAND_HZ),
            ltas_nffts=list(LTAS_NFFTS),
            whitener=dict(n_fft=WHITENER_NFFT, octave=WHITENER_OCTAVE, floor_db=WHITENED_FLOOR_DB),
            texture_nfft=TEXTURE_NFFT,
            hop_ratio=HOP_RATIO,
            modulation_band_hz=list(MOD_BAND_HZ),
            render_seed=RENDER_SEED,
            d_scales=list(D_SCALES),
            sigma_scales=list(SIGMA_SCALES),
        ),
        candidates={k: CANDIDATE_UNITS[k] for k in CANDIDATE_NAMES},
        supports=[],
        rows=[],
    )
    spectro_example: dict[str, Any] | None = None
    for rig in rigs:
        c3 = RE.read_candidate_export(C3_EXPORTS[rig])
        payload.setdefault("c3_parameters", {})[rig] = {
            k: float(c3.summary["parameters"][k]) for k in ("lam", "sigma", "d_scalar")
        }
        scored_list: list[Support] = []
        if rig == "dregon":
            for rec, start, dur in DREGON_SCORED:
                scored_list.append(
                    Support(rig, rec, start, dur, "cruise", "scored", "frozen manifest v2")
                )
        else:
            for rec, start, dur, regime in MICHAELS_SCORED:
                scored_list.append(
                    Support(rig, rec, start, dur, regime, "scored", "frozen evaluator resolution")
                )
        if supports_per_rig is not None:
            scored_list = scored_list[: int(supports_per_rig)]
        for scored in scored_list:
            rec_obj = C.load_recording(DATASET[rig], scored.recording, None, RAW_KEY[rig])
            in_regime = scored.regime != "ramp"
            found = disjoint_segment(rec_obj, rig, scored, min_seconds=1.0, in_regime=in_regime)
            ref, ref_detail = (None, None) if found is None else found
            real_clip = load(scored, rps_key=RAW_KEY[rig], channels=None)
            ref_clip = None if ref is None else load(ref, rps_key=RAW_KEY[rig], channels=None)
            second = None
            if rig == "michaels" and scored.regime == "cruise":
                rec2, start2 = MICHAELS_SECOND_REAL
                sup2 = Support(
                    rig,
                    rec2,
                    start2,
                    scored.duration_s,
                    "cruise",
                    "second_real",
                    "FLY125 cruise window, real-vs-real oracle only",
                )
                second = np.asarray(
                    load(sup2, rps_key=RAW_KEY[rig], channels=None).audio, dtype=np.float64
                )
            entry: dict[str, Any] = dict(
                scored=scored.as_dict(),
                oracle_reference=None if ref is None else ref.as_dict(),
                oracle_match=ref_detail,
                second_real=None
                if second is None
                else dict(recording=MICHAELS_SECOND_REAL[0], start_s=MICHAELS_SECOND_REAL[1]),
                guard_seconds=GUARD_S,
            )
            payload["supports"].append(entry)
            real = np.asarray(real_clip.audio, dtype=np.float64)
            rates = np.atleast_2d(np.asarray(real_clip.rps, dtype=np.float64)).mean(axis=1)
            entry["mean_rps"] = [float(v) for v in rates]
            w_freqs, w = whitener(real[MIC])
            t0 = time.time()
            arms = build_arms(
                rig=rig,
                scored=scored,
                real_clip=real_clip,
                ref_clip=ref_clip,
                second_real=second,
                c3_summary=c3.summary,
                ladder=ladder,
                n_mics=n_mics,
            )
            entry["render_seconds"] = float(time.time() - t0)
            for arm in arms:
                real_side = real if arm.real_override is None else arm.real_override
                vals = candidates(real_side[MIC], arm.audio[MIC], w_freqs=w_freqs, w=w, rates=rates)
                row: dict[str, Any] = dict(
                    rig=rig,
                    regime=scored.regime,
                    support=scored.window.key,
                    arm=arm.name,
                    kind=arm.kind,
                    detail=arm.detail,
                )
                for name in CANDIDATE_NAMES:
                    row[name] = vals[name]
                payload["rows"].append(row)
                if spectro_example is None and rig == "dregon" and arm.name == "current_best":
                    spectro_example = dict(
                        support=scored.window.key,
                        real=real[MIC],
                        arm=arm.audio[MIC],
                        w_freqs=w_freqs,
                        w=w,
                    )
            print(
                f"[{rig}] {scored.window.key}: {len(arms)} arms, {entry['render_seconds']:.1f} s renders",
                flush=True,
            )
    payload["_spectro_example"] = spectro_example
    return payload


# ── the verdicts ────────────────────────────────────────────────────────────


def _val(row: dict[str, Any], name: str) -> float | None:
    v = row.get(name, {}).get("value")
    return None if v is None else float(v)


def _rho(pts: list[tuple[float, float]]) -> float | None:
    """Spearman rho of a scale/value ladder, or ``None`` below three points.

    Three is the minimum a rank correlation can say anything with, and the
    sigma degradation branch (x1, x3, x10) has exactly three.
    """
    if len(pts) < 3:
        return None
    rho = float(np.asarray(sstats.spearmanr([p[0] for p in pts], [p[1] for p in pts])[0]))
    return rho if np.isfinite(rho) else None


def ladder_points(
    rows: list[dict[str, Any]],
    *,
    name: str,
    tag: str,
    scales: list[float],
    support: str | None = None,
) -> list[tuple[float, float]]:
    """``[(scale, value)]`` of one ladder, averaged over supports unless one is named."""
    out: list[tuple[float, float]] = []
    for s in scales:
        arm = "c3" if s == 1.0 else f"{tag}_x{s:g}"
        sel = [r for r in rows if r["arm"] == arm and (support is None or r["support"] == support)]
        vals = [v for v in (_val(r, name) for r in sel) if v is not None]
        if vals:
            out.append((float(s), float(np.mean(vals))))
    return out


def summarise(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["rows"]
    groups = sorted({(r["rig"], r["regime"]) for r in rows}) + sorted(
        {(r["rig"], "__pooled__") for r in rows}
    )
    out: dict[str, Any] = dict(per_group=[], ladders=[])
    d_scales = list(payload["front_end"]["d_scales"])
    s_scales = list(payload["front_end"]["sigma_scales"])
    for rig, regime in groups:
        rr = [
            r for r in rows if r["rig"] == rig and (regime == "__pooled__" or r["regime"] == regime)
        ]
        supports = sorted({r["support"] for r in rr})
        for name in CANDIDATE_NAMES:

            def arm_values(arm: str, rows_=rr, name_=name) -> list[float]:
                return [
                    v for v in (_val(r, name_) for r in rows_ if r["arm"] == arm) if v is not None
                ]

            oracle = arm_values("oracle")
            oracle_cross = arm_values("oracle_cross_recording")
            oracle_split = arm_values("oracle_split_half")
            best = arm_values("current_best")
            c3 = arm_values("c3")
            spread = float(np.std(oracle, ddof=1)) if len(oracle) > 1 else None
            floor = float(np.mean(oracle)) if oracle else None
            best_mean = float(np.mean(best)) if best else None
            c3_mean = float(np.mean(c3)) if c3 else None
            # ladder response, per support (so a rho is not an artefact of pooling)
            rho: dict[str, list[float]] = dict(d=[], d_up=[], sigma=[], sigma_up=[])
            minima: dict[str, float | None] = dict(d=None, sigma=None)
            spans: dict[str, float | None] = dict(d=None, sigma=None)
            for tag, scales in (("d", d_scales), ("sigma", s_scales)):
                for support in supports:
                    pts = ladder_points(rr, name=name, tag=tag, scales=scales, support=support)
                    full = _rho(pts)
                    if full is not None:
                        rho[tag].append(full)
                    # the DEGRADATION branch: scales at or above the export's own
                    # value. A V-shaped ladder is not monotone over the whole
                    # range, and the branch away from the minimum is what a
                    # threshold has to be monotone on.
                    up = _rho([p for p in pts if p[0] >= 1.0])
                    if up is not None:
                        rho[f"{tag}_up"].append(up)
                # the POOLED ladder's own minimum and dynamic range: a median of
                # per-support argmins can land on a scale no support prefers
                pooled = ladder_points(rr, name=name, tag=tag, scales=scales)
                if pooled:
                    minima[tag] = float(min(pooled, key=lambda p: p[1])[0])
                    spans[tag] = float(max(p[1] for p in pooled) - min(p[1] for p in pooled))
            out["per_group"].append(
                dict(
                    rig=rig,
                    regime=regime,
                    candidate=name,
                    units=CANDIDATE_UNITS[name],
                    oracle_floor=floor,
                    oracle_spread=spread,
                    oracle_min=float(np.min(oracle)) if oracle else None,
                    oracle_max=float(np.max(oracle)) if oracle else None,
                    oracle_cross_recording=float(np.mean(oracle_cross)) if oracle_cross else None,
                    oracle_split_half=float(np.mean(oracle_split)) if oracle_split else None,
                    oracle_split_half_spread=(
                        float(np.std(oracle_split, ddof=1)) if len(oracle_split) > 1 else None
                    ),
                    current_best=best_mean,
                    c3=c3_mean,
                    best_minus_oracle=(
                        None if (best_mean is None or floor is None) else float(best_mean - floor)
                    ),
                    c3_minus_oracle=(
                        None if (c3_mean is None or floor is None) else float(c3_mean - floor)
                    ),
                    separation_ratio=(
                        None
                        if (best_mean is None or floor is None or not spread)
                        else float((best_mean - floor) / spread)
                    ),
                    spearman_rho_d=float(np.median(rho["d"])) if rho["d"] else None,
                    spearman_rho_d_branch=float(np.median(rho["d_up"])) if rho["d_up"] else None,
                    spearman_rho_d_branch_min=float(np.min(rho["d_up"])) if rho["d_up"] else None,
                    spearman_rho_sigma=float(np.median(rho["sigma"])) if rho["sigma"] else None,
                    spearman_rho_sigma_branch=(
                        float(np.median(rho["sigma_up"])) if rho["sigma_up"] else None
                    ),
                    ladder_min_d_scale=minima["d"],
                    ladder_min_sigma_scale=minima["sigma"],
                    ladder_span_d=spans["d"],
                    ladder_span_sigma=spans["sigma"],
                    gap_closure_current_best=(
                        None
                        if (
                            best_mean is None
                            or floor is None
                            or c3_mean is None
                            or c3_mean == floor
                        )
                        else float((c3_mean - best_mean) / (c3_mean - floor))
                    ),
                    n_supports=len(oracle),
                )
            )
            for tag, scales in (("d", d_scales), ("sigma", s_scales)):
                for s in scales:
                    arm = "c3" if s == 1.0 else f"{tag}_x{s:g}"
                    vals = arm_values(arm)
                    if not vals:
                        continue
                    out["ladders"].append(
                        dict(
                            rig=rig,
                            regime=regime,
                            candidate=name,
                            ladder=tag,
                            scale=float(s),
                            mean=float(np.mean(vals)),
                            std=float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                            n=len(vals),
                        )
                    )
    return out


# ── figures ─────────────────────────────────────────────────────────────────


def figure_bars(
    payload: dict[str, Any], summary: dict[str, Any], fig_dir: Path, out_dir: Path
) -> list[str]:
    per = [r for r in summary["per_group"] if r["regime"] != "__pooled__"]
    groups = sorted({(r["rig"], r["regime"]) for r in per})
    labels = [f"{rig}\n{regime}" for rig, regime in groups]
    fig, axes = plt.subplots(
        1, len(CANDIDATE_NAMES), figsize=(3.3 * len(CANDIDATE_NAMES), 4.0), squeeze=False
    )
    width = 0.21
    series = (
        ("oracle_split_half", "C8", "oracle (split half, half length)"),
        ("oracle_floor", "C2", "oracle (speed-matched disjoint)"),
        ("current_best", "C0", "current best"),
        ("c3", "C3", "C3 (known bad)"),
    )
    for ci, name in enumerate(CANDIDATE_NAMES):
        ax = axes[0][ci]
        xs = np.arange(len(groups))
        for oi, (field, col, lab) in enumerate(series):
            vals, errs = [], []
            for rig, regime in groups:
                row = next(
                    r for r in per if (r["rig"], r["regime"], r["candidate"]) == (rig, regime, name)
                )
                vals.append(np.nan if row[field] is None else row[field])
                spread_field = (
                    "oracle_split_half_spread" if field == "oracle_split_half" else "oracle_spread"
                )
                errs.append(
                    row[spread_field]
                    if (field.startswith("oracle") and row.get(spread_field))
                    else 0.0
                )
            ax.bar(xs + (oi - 1.5) * width, vals, width, yerr=errs, capsize=2, color=col, label=lab)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(name, fontsize=9)
        ax.set_ylabel(CANDIDATE_UNITS[name].split(",")[0], fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, axis="y")
    axes[0][0].legend(fontsize=6)
    fig.suptitle(
        "candidate proxies per rig and regime: oracle floor (error bar = spread across supports), "
        "current-best synthetic, C3",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    name = "criteria_proxy_bars.png"
    for d in (fig_dir, out_dir):
        fig.savefig(d / name, dpi=140)
    plt.close(fig)
    return [name]


def figure_ladders(
    payload: dict[str, Any], summary: dict[str, Any], fig_dir: Path, out_dir: Path
) -> list[str]:
    lad = summary["ladders"]
    if not lad:
        return []
    names = []
    for tag, label in (("d", "D scale"), ("sigma", "sigma scale")):
        rows = [r for r in lad if r["ladder"] == tag and r["regime"] == "cruise"]
        if not rows:
            continue
        rigs = sorted({r["rig"] for r in rows})
        fig, axes = plt.subplots(
            len(rigs),
            len(CANDIDATE_NAMES),
            figsize=(2.9 * len(CANDIDATE_NAMES), 3.0 * len(rigs)),
            squeeze=False,
        )
        for ri, rig in enumerate(rigs):
            for ci, cand in enumerate(CANDIDATE_NAMES):
                ax = axes[ri][ci]
                sel = sorted(
                    [r for r in rows if r["rig"] == rig and r["candidate"] == cand],
                    key=lambda r: r["scale"],
                )
                if sel:
                    ax.errorbar(
                        [r["scale"] for r in sel],
                        [r["mean"] for r in sel],
                        yerr=[r["std"] or 0.0 for r in sel],
                        marker="o",
                        ms=4,
                        lw=1.2,
                        capsize=2,
                        color="C1",
                    )
                per = next(
                    (
                        p
                        for p in summary["per_group"]
                        if (p["rig"], p["regime"], p["candidate"]) == (rig, "cruise", cand)
                    ),
                    None,
                )
                if per is not None and per["oracle_floor"] is not None:
                    ax.axhline(per["oracle_floor"], color="C2", ls="--", lw=1.0, label="oracle")
                if per is not None and per["current_best"] is not None:
                    ax.axhline(
                        per["current_best"], color="C0", ls=":", lw=1.0, label="current best"
                    )
                ax.set_xscale("log")
                ax.grid(alpha=0.25)
                ax.tick_params(labelsize=7)
                if ri == 0:
                    ax.set_title(cand, fontsize=8)
                if ci == 0:
                    ax.set_ylabel(f"{rig}", fontsize=8)
                ax.set_xlabel(label, fontsize=7)
        axes[0][0].legend(fontsize=6)
        fig.suptitle(
            f"CRUISE degradation ladder against the {label} (C3 base, rendered on the real carrier)",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        name = f"criteria_proxy_ladder_{tag}.png"
        for d in (fig_dir, out_dir):
            fig.savefig(d / name, dpi=140)
        plt.close(fig)
        names.append(name)
    return names


def figure_spectrograms(example: dict[str, Any] | None, fig_dir: Path, out_dir: Path) -> list[str]:
    if example is None:
        return []
    real, arm = example["real"], example["arm"]
    w_freqs, w = example["w_freqs"], example["w"]
    freqs, pr = spectrogram(real, TEXTURE_NFFT)
    _, pa = spectrogram(arm, TEXTURE_NFFT)
    n = min(pr.shape[0], pa.shape[0])
    wg = whitener_on(w_freqs, w, freqs)
    m = band_mask(freqs)
    frame_rate = SR / float(max(TEXTURE_NFFT // HOP_RATIO, 1))
    times = np.arange(n) / frame_rate
    panels = [
        ("real, unwhitened", log_mag(pr[:n], w=None, floor_db=None)),
        ("current best, unwhitened", log_mag(pa[:n], w=None, floor_db=None)),
        ("real, whitened", log_mag(pr[:n], w=wg, floor_db=WHITENED_FLOOR_DB)),
        ("current best, whitened", log_mag(pa[:n], w=wg, floor_db=WHITENED_FLOOR_DB)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.4), squeeze=False)
    for ax, (title, db) in zip(axes.ravel(), panels, strict=True):
        z = db[:, m]
        whit = "whitened" in title
        vmax = float(np.percentile(z, 99.5))
        vmin = vmax - (30.0 if whit else 45.0)
        im = ax.pcolormesh(times, freqs[m], z.T, vmin=vmin, vmax=vmax, shading="auto", cmap="magma")
        ax.set_yscale("log")
        ax.set_ylim(max(BAND_HZ[0], 40.0), BAND_HZ[1])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel("Hz", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, pad=0.01).ax.tick_params(labelsize=6)
    fig.suptitle(
        f"NFFT {TEXTURE_NFFT} log spectrogram, {example['support']}\n"
        f"whitened panels divide BOTH clips by the real clip's 1/6-octave LTAS and floor at "
        f"{WHITENED_FLOOR_DB:g} dB",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    name = "criteria_proxy_spectrogram.png"
    for d in (fig_dir, out_dir):
        fig.savefig(d / name, dpi=140)
    plt.close(fig)
    return [name]


# ── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--fig-dir", type=Path, default=FIG_DEFAULT)
    ap.add_argument("--rigs", default="dregon,michaels")
    ap.add_argument("--supports-per-rig", type=int, default=None)
    ap.add_argument("--no-ladder", action="store_true")
    ap.add_argument("--mics", type=int, default=8)
    ap.add_argument(
        "--redraw",
        action="store_true",
        help="recompute nothing: re-summarise and redraw from the existing proxy.json",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    example: dict[str, Any] | None = None
    prior_figures: list[str] = []
    if args.redraw:
        payload = json.loads((out_dir / "proxy.json").read_text())
        payload.pop("summary", None)
        prior_figures = list(payload.pop("figures", []) or [])
    else:
        payload = run(
            rigs=[v for v in str(args.rigs).replace(",", " ").split()],
            supports_per_rig=args.supports_per_rig,
            ladder=not args.no_ladder,
            n_mics=int(args.mics),
        )
        example = payload.pop("_spectro_example")
    payload["summary"] = summarise(payload)
    figures = figure_bars(payload, payload["summary"], fig_dir, out_dir)
    figures += figure_ladders(payload, payload["summary"], fig_dir, out_dir)
    figures += figure_spectrograms(example, fig_dir, out_dir)
    # a redraw does not re-render, so a figure it cannot rebuild keeps its name
    figures += [f for f in prior_figures if f not in figures]
    payload["figures"] = figures
    payload["provenance"] = dict(
        git_head=git_head(),
        import_provenance=RE.import_provenance(),
        argv=vars(args) | dict(out=str(args.out), fig_dir=str(args.fig_dir)),
        wall_seconds=float(time.time() - t0),
    )
    RE.write_json(out_dir / "proxy.json", payload)
    print(f"wrote {out_dir / 'proxy.json'}")
    print("figures:", ", ".join(figures))
    print(f"total wall {payload['provenance']['wall_seconds']:.1f} s")


if __name__ == "__main__":
    main()
