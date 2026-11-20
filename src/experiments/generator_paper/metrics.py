"""Real-vs-generated fidelity metrics for the generator paper.

:func:`evaluate_pair` compares one REFERENCE recording ``real (M, T)`` with one
candidate ``generated (M, T)`` rendered on the same rotor label ``rps (R, T)``
(rev/s on the audio grid) and returns::

    {"summary":  {key: finite float},            # flat, one row per pair
     "details":  {...},                          # JSON-serialisable diagnostics
     "metadata": {...}}                          # definitions, normalisation, versions

Signed differences are ``generated - real`` throughout. ``summary`` holds only
finite values; a key whose value is undefined for this pair (empty support,
silent candidate, single microphone, ...) is left out of ``summary`` and listed
in ``details["undefined"]`` with the reason. ``metadata["summary_keys"]`` lists
every key the function can emit, so a missing key is never confused with a
missing implementation.

Three groups of quantities:

1. **Long-term spectrum** (defined here).
   ``lsd_db`` is a true log-spectral distance — the RMS over frequency of the
   10 log10 power ratio of the two Welch PSDs, NOT the mean absolute dB
   difference that older code in this repository calls "LSD".
   ``third_octave_mae_db`` is the mean absolute difference of third-octave
   band levels.
2. **Repository diagnostics** reused from
   :mod:`experiments.noise_model.similarity` (custom instruments of this
   project, not literature-standard metrics): line/floor dB differences,
   order-tracked line levels (all valid orders, and the subset VISIBLE in the
   reference), frame-level distributions and modulation spectra, spatial
   level spread / coherence, Itakura-Saito divergence, and a Fréchet distance
   of third-octave frame-level vectors.
3. **Nominal-playback psychoacoustic descriptors** from MoSQITo: Aures
   free-field sharpness computed from Zwicker stationary loudness
   (ISO 532-1:2017 method 1). Our PCM carries no pascal calibration, so both
   signals are scaled by ONE common gain that puts the reference's multi-mic
   RMS at a nominal 80 dB SPL playback level; the values describe that
   nominal playback, not the aircraft's measured SPL or absolute loudness.
"""

from __future__ import annotations

import math
from importlib.metadata import version
from typing import Any

import numpy as np
from scipy.signal import resample_poly, welch

from experiments.noise_model import similarity as sim

try:
    from mosqito.sound_level_meter import noct_spectrum
    from mosqito.sq_metrics import sharpness_din_from_loudness
    from mosqito.sq_metrics.loudness.loudness_zwst._calc_slopes import _calc_slopes
    from mosqito.sq_metrics.loudness.loudness_zwst._main_loudness import _main_loudness
    from mosqito.utils import amp2db
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise ModuleNotFoundError(
        "experiments.generator_paper.metrics needs MoSQITo (pinned mosqito==1.2.1 in "
        "pyproject.toml) for the Aures sharpness descriptor; run `uv sync`."
    ) from exc

__all__ = [
    "LSD_BAND_HZ",
    "NOMINAL_SPL_DB",
    "THIRD_OCTAVE_CENTRES_HZ",
    "THIRD_OCTAVE_NOMINAL",
    "VISIBLE_PROMINENCE_DB",
    "evaluate_pair",
]

# ── long-term spectrum ────────────────────────────────────────────────────

#: Welch PSD: periodic Hann, 4096 points (3.9 Hz at 16 kHz), hop 1024 (75 % overlap).
WELCH_NFFT, WELCH_HOP = 4096, 1024
#: Inclusive frequency band of ``lsd_db``.
LSD_BAND_HZ = (125.0, 7200.0)
#: Numerical floor added to BOTH PSDs before the log: this fraction (-100 dB) of
#: the reference's mean PSD over the LSD band, pooled over mics. Relative, so a
#: common gain on both signals leaves every dB metric unchanged; only bins more
#: than 100 dB below the reference's mean level are affected.
PSD_FLOOR_REL = 1e-10
#: Base-2 exact third-octave centres 1000 * 2^(k/3), k = -9..8 (nominal
#: 125 Hz - 6.3 kHz); band edges centre * 2^(+-1/6), i.e. 111 Hz - 7.13 kHz.
_THIRD_K = np.arange(-9, 9)
THIRD_OCTAVE_CENTRES_HZ = 1000.0 * 2.0 ** (_THIRD_K / 3.0)
THIRD_OCTAVE_NOMINAL = (
    "125", "160", "200", "250", "315", "400", "500", "630", "800",
    "1000", "1250", "1600", "2000", "2500", "3150", "4000", "5000", "6300",
)  # fmt: skip
_THIRD_EDGES = np.stack(
    [THIRD_OCTAVE_CENTRES_HZ * 2.0 ** (-1 / 6), THIRD_OCTAVE_CENTRES_HZ * 2.0 ** (1 / 6)], axis=1
)

# ── order tracking: reference visibility ──────────────────────────────────

#: An order cell (mic, rotor, k) is REFERENCE-VISIBLE when the reference's
#: frame-mean prominence (similarity's peak-over-side-median statistic) is at
#: least this. Custom threshold: on 8 mics x 4 s of white noise with rotor
#: labels at 20/50/80 rev/s, the same statistic's 99.9th percentile is
#: 2.4/4.8/5.5 dB, so 6 dB sits above the noise-only bias.
VISIBLE_PROMINENCE_DB = 6.0

# ── nominal-playback psychoacoustics ──────────────────────────────────────

#: The reference's RMS (pooled over all mics and samples) is mapped to this SPL.
NOMINAL_SPL_DB = 80.0
P_REF_PA = 2e-5
#: MoSQITo's Zwicker loudness wants fs >= 48 kHz; we upsample (polyphase)
#: ourselves instead of letting it FFT-resample and print a warning.
PSY_FS = 48000
#: Below this loudness the Aures weighting's N / ln(0.05 N + 1) term and the
#: S = ... / N normalisation are ill-conditioned; MoSQITo's own time-varying
#: path zeroes sharpness there. We report it UNDEFINED instead of zero.
MIN_LOUDNESS_SONE = 0.1
_ZWICKER_LOW_BAND_LIMIT_DB = 120.0  # ISO 532-1:2017, A.3; first 11 bands, 25-250 Hz.

_ORDER_GROUP_NAMES = tuple(f"k{lo}-{hi}" for lo, hi in sim.ORDER_GROUPS)


# ── helpers ───────────────────────────────────────────────────────────────


def _as_2d(x: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D (channels, time); got shape {a.shape}")
    if not np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} holds non-finite samples")
    return a


def _mic_mean(x: np.ndarray) -> np.ndarray:
    """``(R, K)`` mean of ``x (M, R, K)`` over mics; NaN where every mic is NaN
    (orders outside the band), without numpy's empty-slice warning."""
    n = np.isfinite(x).sum(axis=0)
    total = np.where(np.isfinite(x), x, 0.0).sum(axis=0)
    return np.where(n > 0, total / np.maximum(n, 1), np.nan)


def _jsonable(x: Any) -> Any:
    """Arrays/scalars -> nested lists of float/int; non-finite floats -> None."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


def _db(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(x)


# ── 1. long-term spectrum ─────────────────────────────────────────────────


def _welch_psd(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """``(F,)`` frequencies and ``(M, F)`` one-sided PSD (power / Hz)."""
    return welch(
        x,
        fs=sr,
        window="hann",  # scipy's get_window default is periodic (sym=False)
        nperseg=WELCH_NFFT,
        noverlap=WELCH_NFFT - WELCH_HOP,
        nfft=WELCH_NFFT,
        detrend=False,  # pyright: ignore[reportArgumentType]
        scaling="density",
        axis=-1,
    )


def _spectrum_metrics(
    real: np.ndarray, gen: np.ndarray, sr: int
) -> tuple[dict[str, float], dict[str, Any]]:
    freqs, p_r = _welch_psd(real, sr)
    _, p_g = _welch_psd(gen, sr)
    band = (freqs >= LSD_BAND_HZ[0]) & (freqs <= LSD_BAND_HZ[1])
    if not band.any():
        raise ValueError(f"sr={sr} leaves no Welch bin in {LSD_BAND_HZ} Hz")
    ref_mean = float(p_r[:, band].mean())
    if not ref_mean > 0.0:
        raise ValueError("reference is silent over the LSD band; no relative PSD floor exists")
    floor = PSD_FLOOR_REL * ref_mean
    lr = _db(p_r + floor)
    lg = _db(p_g + floor)
    d = (lg - lr)[:, band]  # (M, Fb)
    lsd_mic = np.sqrt(np.mean(d**2, axis=1))
    bias_mic = d.mean(axis=1)

    df = freqs[1] - freqs[0]
    n_third = len(THIRD_OCTAVE_CENTRES_HZ)
    tb_r = np.empty((real.shape[0], n_third))
    tb_g = np.empty_like(tb_r)
    bins = np.zeros(n_third, dtype=int)
    for j, (lo, hi) in enumerate(_THIRD_EDGES):
        sel = (freqs >= lo) & (freqs < hi)
        bins[j] = int(sel.sum())
        if bins[j] == 0:
            raise ValueError(f"third-octave band {lo:.1f}-{hi:.1f} Hz holds no Welch bin")
        tb_r[:, j] = (p_r[:, sel] + floor).sum(axis=1) * df
        tb_g[:, j] = (p_g[:, sel] + floor).sum(axis=1) * df
    d3 = _db(tb_g) - _db(tb_r)  # (M, B3)

    summary = dict(
        lsd_db=float(lsd_mic.mean()),
        psd_mean_diff_db=float(bias_mic.mean()),
        third_octave_mae_db=float(np.abs(d3).mean()),
        third_octave_signed_diff_db=float(d3.mean()),
    )
    details = dict(
        lsd_db_per_mic=lsd_mic,
        psd_mean_diff_db_per_mic=bias_mic,
        lsd_bins=int(band.sum()),
        psd_floor=floor,
        third_octave=dict(
            nominal_hz=list(THIRD_OCTAVE_NOMINAL),
            centre_hz=THIRD_OCTAVE_CENTRES_HZ,
            edges_hz=_THIRD_EDGES,
            bins=bins,
            real_level_db=_db(tb_r),
            generated_level_db=_db(tb_g),
            abs_diff_db_per_band=np.abs(d3).mean(axis=0),
            signed_diff_db_per_band=d3.mean(axis=0),
        ),
    )
    return summary, details


# ── 2. order-tracked line levels (all valid / reference-visible) ──────────


def _order_metrics(
    real: np.ndarray, gen: np.ndarray, rps: np.ndarray, sr: int
) -> tuple[dict[str, float], dict[str, Any]]:
    """The :func:`similarity.order_level_error` statistic, computed once and
    aggregated twice: over every valid order (identical to that function's
    summary) and over the cells visible in the reference."""
    freqs = np.fft.rfftfreq(sim.FINE_NFFT, 1.0 / sr)
    pr = np.abs(sim._stft(real, sim.FINE_NFFT, sim.FINE_HOP)) ** 2
    pg = np.abs(sim._stft(gen, sim.FINE_NFFT, sim.FINE_HOP)) ** 2
    flab = sim._frame_labels(rps, pr.shape[1], sim.FINE_NFFT, sim.FINE_HOP)
    peak_r, prom_r = sim._order_levels(pr, freqs, flab)
    peak_g, prom_g = sim._order_levels(pg, freqs, flab)
    valid = np.isfinite(peak_r)  # (M, R, K); NaN marks orders outside the band
    visible = valid & (prom_r >= VISIBLE_PROMINENCE_DB)
    d_peak = peak_g - peak_r
    d_prom = prom_g - prom_r
    masked_peak = np.where(visible, d_peak, np.nan)
    masked_prom = np.where(visible, d_prom, np.nan)
    agg = dict(
        all=(sim._group_errors(d_peak), sim._group_errors(d_prom)),
        visible=(sim._group_errors(masked_peak), sim._group_errors(masked_prom)),
    )
    summary: dict[str, float] = {}
    for scope, (pk, pm) in agg.items():
        summary[f"order_{scope}__peak_abs_db"] = float(pk["mean_abs_db"])
        summary[f"order_{scope}__prominence_abs_db"] = float(pm["mean_abs_db"])
        for g, name in enumerate(_ORDER_GROUP_NAMES):
            summary[f"order_{scope}__peak_abs_db_{name}"] = float(pk["abs_db"][g])
            summary[f"order_{scope}__peak_signed_db_{name}"] = float(pk["signed_db"][g])
            summary[f"order_{scope}__prominence_abs_db_{name}"] = float(pm["abs_db"][g])
    cov_valid = np.zeros(len(sim.ORDER_GROUPS), dtype=int)
    cov_visible = np.zeros_like(cov_valid)
    for g, (lo, hi) in enumerate(sim.ORDER_GROUPS):
        cov_valid[g] = int(valid[..., lo - 1 : hi].sum())
        cov_visible[g] = int(visible[..., lo - 1 : hi].sum())
    details = dict(
        groups=list(_ORDER_GROUP_NAMES),
        visible_prominence_db=VISIBLE_PROMINENCE_DB,
        valid_cells_per_group=cov_valid,
        visible_cells_per_group=cov_visible,
        visible_fraction_per_rotor_order=visible.mean(axis=0),  # (R, K)
        real_peak_db=_mic_mean(peak_r),
        generated_peak_db=_mic_mean(peak_g),
        real_prominence_db=_mic_mean(prom_r),
        generated_prominence_db=_mic_mean(prom_g),
        all=dict(peak=agg["all"][0], prominence=agg["all"][1]),
        visible=dict(peak=agg["visible"][0], prominence=agg["visible"][1]),
    )
    return summary, details


# ── 3. nominal-playback psychoacoustics (MoSQITo) ─────────────────────────


def _zwicker_aures(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-mic loudness, sharpness and maximum 25-250 Hz third-octave level.

    Reuse MoSQITo 1.2.1's exact stationary front end and numerical kernels.
    Its public wrapper raises above the ISO A.3 input domain; such a channel
    has no defined loudness/sharpness here, not a capped or rescaled value.
    Other numerical errors still propagate.
    """
    g = math.gcd(int(sr), PSY_FS)
    up = resample_poly(x, PSY_FS // g, int(sr) // g, axis=-1)
    n = np.full(x.shape[0], np.nan)
    s = np.full(x.shape[0], np.nan)
    low_band_max = np.empty(x.shape[0])
    for m in range(x.shape[0]):
        third, _ = noct_spectrum(up[m], PSY_FS, fmin=24, fmax=12600)
        levels = amp2db(third, ref=P_REF_PA)  # pyright: ignore[reportArgumentType]
        low_band_max[m] = float(np.max(levels[:11]))
        if low_band_max[m] > _ZWICKER_LOW_BAND_LIMIT_DB:
            continue
        loud, spec = _calc_slopes(_main_loudness(levels, "free"))
        n[m] = float(loud)
        if n[m] >= MIN_LOUDNESS_SONE:
            s[m] = float(sharpness_din_from_loudness(loud, spec, weighting="aures"))
    return n, s, low_band_max


def _psycho_metrics(
    real: np.ndarray, gen: np.ndarray, sr: int
) -> tuple[dict[str, float], dict[str, Any]]:
    ref_rms = float(np.sqrt(np.mean(np.square(real, dtype=np.float64))))
    gain = P_REF_PA * 10.0 ** (NOMINAL_SPL_DB / 20.0) / ref_rms
    n_r, s_r, low_r = _zwicker_aures(real * gain, sr)
    n_g, s_g, low_g = _zwicker_aures(gen * gain, sr)
    domain_valid = np.isfinite(n_r) & np.isfinite(n_g)
    # A failed channel invalidates the pair-level comparison; do not average
    # only the surviving microphones and thereby hide an excessive output.
    both = np.isfinite(s_r) & np.isfinite(s_g) & bool(domain_valid.all())
    loud_ok = (n_r >= MIN_LOUDNESS_SONE) & bool(domain_valid.all())
    p = "nominal80__"
    nan = float("nan")
    summary = {
        f"{p}zwicker_loudness_ref_sone": float(n_r.mean()),
        f"{p}zwicker_loudness_gen_sone": float(n_g.mean()),
        f"{p}zwicker_loudness_rel_diff": (
            float(np.mean(n_g[loud_ok] / n_r[loud_ok] - 1.0)) if loud_ok.any() else nan
        ),
        f"{p}aures_sharpness_ref_acum": float(np.mean(s_r[both])) if both.any() else nan,
        f"{p}aures_sharpness_gen_acum": float(np.mean(s_g[both])) if both.any() else nan,
        f"{p}aures_sharpness_signed_diff_acum": (
            float(np.mean(s_g[both] - s_r[both])) if both.any() else nan
        ),
        f"{p}aures_sharpness_abs_diff_acum": (
            float(np.mean(np.abs(s_g[both] - s_r[both]))) if both.any() else nan
        ),
        f"{p}aures_sharpness_rel_diff": (
            float(np.mean(s_g[both] / s_r[both] - 1.0)) if both.any() else nan
        ),
    }
    details = dict(
        common_gain_pa_per_unit=gain,
        reference_rms_pcm=ref_rms,
        zwicker_loudness_ref_sone_per_mic=n_r,
        zwicker_loudness_gen_sone_per_mic=n_g,
        aures_sharpness_ref_acum_per_mic=s_r,
        aures_sharpness_gen_acum_per_mic=s_g,
        mics_with_defined_sharpness=int(both.sum()),
        mics_with_defined_loudness_ratio=int(loud_ok.sum()),
        zwicker_domain_valid_per_mic=domain_valid,
        zwicker_low_band_max_ref_db_per_mic=low_r,
        zwicker_low_band_max_gen_db_per_mic=low_g,
        zwicker_low_band_limit_db=_ZWICKER_LOW_BAND_LIMIT_DB,
    )
    return summary, details


# ── metadata ──────────────────────────────────────────────────────────────

_REUSED = {
    "line_floor": "line_floor_lsd",
    "level_dist": "level_distribution",
    "spatial": "spatial",
    "itakura_saito": "itakura_saito",
    "spectro_temporal": "spectro_temporal",
}
_REUSED_KEYS = {
    "line_floor": (
        "line_abs_db", "floor_abs_db", "line_signed_db", "floor_signed_db", "smear_fraction",
    ),
    "level_dist": (
        "w1_db", "w1_centered_db", "sd_abs_diff_db", "sd_signed_diff_db", "spread_abs_diff_db",
        "spread_signed_diff_db", "mod_abs_abs_db", "mod_norm_abs_db", "mod_norm_signed_db",
    ),
    "spatial": (
        "level_sd_abs_diff_db", "level_sd_signed_diff_db", "msc_abs_diff", "msc_signed_diff",
    ),
    "itakura_saito": ("is_div", "is_symmetric", "is_octave", "is_octave_symmetric"),
    "spectro_temporal": (
        "frechet_db2", "frechet_mean_term_db2", "frechet_cov_term_db2", "comod_frobenius",
        "real_offdiag_corr", "arm_offdiag_corr", "sd_signed_diff_db", "sd_abs_diff_db",
    ),
}  # fmt: skip


def _summary_keys() -> list[str]:
    keys = ["lsd_db", "psd_mean_diff_db", "third_octave_mae_db", "third_octave_signed_diff_db"]
    for scope in ("all", "visible"):
        keys += [f"order_{scope}__peak_abs_db", f"order_{scope}__prominence_abs_db"]
        for name in _ORDER_GROUP_NAMES:
            keys += [
                f"order_{scope}__peak_abs_db_{name}",
                f"order_{scope}__peak_signed_db_{name}",
                f"order_{scope}__prominence_abs_db_{name}",
            ]
    for prefix, names in _REUSED_KEYS.items():
        keys += [f"{prefix}__{k}" for k in names]
    keys += [
        f"nominal80__{k}"
        for k in (
            "zwicker_loudness_ref_sone", "zwicker_loudness_gen_sone", "zwicker_loudness_rel_diff",
            "aures_sharpness_ref_acum", "aures_sharpness_gen_acum",
            "aures_sharpness_signed_diff_acum", "aures_sharpness_abs_diff_acum",
            "aures_sharpness_rel_diff",
        )
    ]  # fmt: skip
    return keys


def _metadata(sr: int) -> dict[str, Any]:
    return dict(
        sign_convention="differences are generated - real",
        sr=sr,
        summary_keys=_summary_keys(),
        undefined_policy=(
            "summary holds finite floats only; a key whose support is empty or whose value "
            "is non-finite is omitted from summary and listed in details['undefined'] with "
            "its reason. No zero is substituted for an undefined value."
        ),
        definitions=dict(
            lsd_db=(
                "Log-spectral distance: per mic sqrt(mean_f (10 log10((Pg+e)/(Pr+e)))^2) over "
                f"Welch bins {LSD_BAND_HZ[0]:g} <= f <= {LSD_BAND_HZ[1]:g} Hz, then the mean "
                f"over mics. Welch: periodic Hann {WELCH_NFFT}, hop {WELCH_HOP}, no detrend, "
                f"density scaling. Floor e = {PSD_FLOOR_REL:g} x the reference's mean PSD over "
                "that band pooled over mics (-100 dB re the reference's mean level), added "
                "to both PSDs. Not the mean-absolute-dB quantity older repo code calls LSD."
            ),
            psd_mean_diff_db="Mean over mics and LSD-band bins of 10 log10((Pg+e)/(Pr+e)) (gain bias).",
            third_octave_mae_db=(
                "Mean over mics and 18 third-octave bands (exact base-2 centres 1000*2^(k/3), "
                "k=-9..8, nominal 125 Hz-6.3 kHz, edges centre*2^(+-1/6) = 111 Hz-7.13 kHz) "
                "of |Lg - Lr|, band level = 10 log10 of the floored Welch PSD summed over "
                "bins lo <= f < hi times the bin width (FFT-bin summation, not an IEC 61260 "
                "filter bank)."
            ),
            third_octave_signed_diff_db="Signed counterpart of third_octave_mae_db.",
            order=(
                "Repository diagnostic (similarity.order_level_error statistic): per fine "
                "frame (Hann 4096, hop 1024) and rotor, peak power within +-0.08 order of "
                "k f_r and its prominence over the median 0.3-0.5 orders away, frame-mean "
                "in dB, orders whose side window stays below 7.9 kHz. order_all__* "
                "aggregates every valid (mic, rotor, k) cell and equals "
                "order_level_error's summary; order_visible__* only cells whose REFERENCE "
                f"prominence >= {VISIBLE_PROMINENCE_DB:g} dB (custom threshold above the "
                "white-noise null). Groups k1-2, 3-10, 11-30, 31-100."
            ),
            line_floor=(
                "Repository diagnostic similarity.line_floor_lsd: mean |dB difference| (NOT "
                "an RMS log-spectral distance despite the function name) of the fine "
                "long-term PSD over line bins (+-0.05 order of the window-mean label) and "
                "over floor cells, per octave band 31.25 Hz-8 kHz, mean over mics/bands."
            ),
            level_dist=(
                "Repository diagnostic similarity.level_distribution: 128 ms / 16 ms-hop "
                "octave-band frame levels; Wasserstein-1 (absolute and mean-removed), sd "
                "and p90-p10 differences, octave modulation-spectrum dB differences 0.5-32 Hz."
            ),
            spatial=(
                "Repository diagnostic similarity.spatial: per octave band, sd over mics "
                "of band level and mean pairwise magnitude-squared coherence; needs >= 2 mics."
            ),
            itakura_saito=(
                "Repository diagnostic similarity.itakura_saito on the long-term fine PSD, "
                "31.25 Hz-7.9 kHz; IS(real|generated), symmetric, linear-bin and octave-balanced."
            ),
            spectro_temporal=(
                "Repository diagnostic similarity.spectro_temporal: Fréchet distance (dB^2) "
                "between Gaussians fitted to per-mic 128 ms third-octave frame-level "
                "vectors (no neural embedding; not FAD), cross-band co-modulation and "
                "per-band level-sd differences."
            ),
            nominal80=(
                "Nominal-playback psychoacoustic descriptors, NOT measured aircraft SPL or "
                "absolute loudness. One common gain g = 20 uPa * 10^(80/20) / RMS(real "
                "pooled over all mics and samples) is applied to real AND generated; each "
                f"mic is polyphase-resampled {sr} Hz -> {PSY_FS} Hz (content stays limited "
                f"to {sr / 2:g} Hz, biasing sharpness low on both sides equally) and "
                "passed to MoSQITo loudness_zwst (Zwicker stationary loudness, ISO "
                "532-1:2017 method 1, free field; ISO 226:1987 contours) then "
                "sharpness_din_from_loudness(weighting='aures'), i.e. Aures' weighting "
                "g(z)=0.078 e^(0.171 z)/z * N/ln(0.05 N + 1) applied to ZWICKER specific "
                "loudness on the DIN 45692 integration grid. Dreier & Vorlaender 2024 "
                "report Aures free-field sharpness; the loudness model under their Aures "
                "value is not established here, so ours (Aures-on-Zwicker) must not be called "
                "identical to theirs, nor to DIN 45692 sharpness (weighting='din'). "
                f"Sharpness is undefined below {MIN_LOUDNESS_SONE:g} sone. Values are "
                "per-mic, then the mean over mics where both sides are defined; *_rel_diff "
                "is mean(gen/ref - 1)."
            ),
        ),
        not_computed=dict(
            ansi_s3_4_2007_n5_loudness=(
                "Dreier 2024 loudness (ANSI S3.4-2007, Glasberg-Moore, N5 percentile): no "
                "maintained Python implementation in MoSQITo 1.2.1; the Zwicker ISO 532-1 "
                "loudness reported here is a different model and is not a substitute."
            ),
            ecma_418_2_tonality=(
                "Dreier 2024 tonality (ECMA-418-2 Sottek hearing model, tu_HMS): MoSQITo "
                "1.2.1 implements ECMA-418-2 loudness and roughness and ECMA-74 TNR/PR, "
                "not ECMA-418-2 tonality; not computed."
            ),
            ecma_418_2_sharpness="Not implemented in MoSQITo 1.2.1; not computed.",
            fad_kad_neural_embeddings="Deliberately excluded (small corpus; no pretrained models).",
        ),
        packages=dict(numpy=version("numpy"), scipy=version("scipy"), mosqito=version("mosqito")),
        constants=dict(
            welch_nfft=WELCH_NFFT,
            welch_hop=WELCH_HOP,
            lsd_band_hz=list(LSD_BAND_HZ),
            psd_floor_rel=PSD_FLOOR_REL,
            visible_prominence_db=VISIBLE_PROMINENCE_DB,
            nominal_spl_db=NOMINAL_SPL_DB,
            p_ref_pa=P_REF_PA,
            psy_fs=PSY_FS,
            min_loudness_sone=MIN_LOUDNESS_SONE,
        ),
    )


# ── undefined-reason bookkeeping ──────────────────────────────────────────


def _reason(key: str, ctx: dict[str, Any]) -> str:
    if key.startswith("line_floor__"):
        stat = key.removeprefix("line_floor__")
        if stat.startswith(("line", "smear")) and ctx["line_bins"] == 0:
            return "no line bins (window-mean label below 5 rev/s or outside 31-8000 Hz)"
        if stat.startswith("floor") and ctx["floor_bins"] == 0:
            return "no floor cells kept in any octave band"
    if key.startswith("order_"):
        scope = "visible" if key.startswith("order_visible") else "valid"
        name = next((n for n in _ORDER_GROUP_NAMES if key.endswith(n)), None)
        counts = ctx[f"order_{scope}"]
        n = int(counts[_ORDER_GROUP_NAMES.index(name)]) if name else int(np.sum(counts))
        if n == 0:
            return f"no {scope} order cells" + (f" in group {name}" if name else "")
    if key.startswith("spatial__") and ctx["n_mics"] < 2:
        return "spatial diagnostics need >= 2 microphones"
    if key.startswith("nominal80__"):
        if ctx["psycho_out_of_domain"]:
            return (
                "ISO 532-1:2017 A.3 / MoSQITo input domain exceeded: a 25-250 Hz "
                "third-octave level is above 120 dB at the fixed common playback gain; "
                "no level cap, renormalization or survivor-microphone average applied"
            )
        return (
            f"Zwicker loudness below {MIN_LOUDNESS_SONE:g} sone on every mic for at least "
            "one side; Aures sharpness / loudness ratio undefined"
        )
    return "instrument returned a non-finite value (e.g. constant frame levels -> undefined correlation)"


# ── entry point ───────────────────────────────────────────────────────────


def evaluate_pair(
    real: np.ndarray, generated: np.ndarray, rps: np.ndarray, *, sr: int = sim.SR
) -> dict[str, Any]:
    """All fidelity metrics of one (reference, candidate) pair.

    ``real`` and ``generated`` are ``(M, T)`` at ``sr``; ``rps`` is ``(R, T)``
    rev/s on the same grid, shared by both sides. Inputs are not modified.
    Raises ``ValueError`` on shape mismatch, non-finite samples, a clip shorter
    than one 4096-point frame or a silent reference."""
    real = _as_2d(real, "real")
    gen = _as_2d(generated, "generated")
    lab = _as_2d(rps, "rps")
    if real.shape != gen.shape:
        raise ValueError(f"real {real.shape} and generated {gen.shape} differ in shape")
    if lab.shape[1] != real.shape[1]:
        raise ValueError(f"rps has {lab.shape[1]} samples, audio has {real.shape[1]}")
    if real.shape[1] < WELCH_NFFT:
        raise ValueError(f"{real.shape[1]} samples is shorter than one {WELCH_NFFT}-point frame")

    cand: dict[str, float] = {}
    details: dict[str, Any] = {}

    s, d = _spectrum_metrics(real, gen, sr)
    cand.update(s)
    details["spectrum"] = d

    s, d = _order_metrics(real, gen, lab, sr)
    cand.update(s)
    details["order"] = d

    lf = sim.line_floor_lsd(real, gen, lab, sr=sr)
    reused: dict[str, dict[str, Any]] = {"line_floor": lf}
    reused["level_dist"] = sim.level_distribution(real, gen, sr=sr)
    if real.shape[0] >= 2:
        reused["spatial"] = sim.spatial(real, gen, sr=sr)
    reused["itakura_saito"] = sim.itakura_saito(real, gen, sr=sr)
    reused["spectro_temporal"] = sim.spectro_temporal(real, gen, sr=sr)
    for prefix, out in reused.items():
        for k in _REUSED_KEYS[prefix]:
            cand[f"{prefix}__{k}"] = float(out["summary"][k])
        details[prefix] = {k: v for k, v in out.items() if k != "summary"}
    if "spatial" not in reused:
        cand.update({f"spatial__{k}": float("nan") for k in _REUSED_KEYS["spatial"]})

    s, d = _psycho_metrics(real, gen, sr)
    cand.update(s)
    details["nominal80_psychoacoustics"] = d

    order_d = details["order"]
    ctx = dict(
        line_bins=int(np.sum(lf["line_bins"])),
        floor_bins=int(np.sum(lf["floor_bins"])),
        order_valid=order_d["valid_cells_per_group"],
        order_visible=order_d["visible_cells_per_group"],
        n_mics=real.shape[0],
        psycho_out_of_domain=not bool(np.all(d["zwicker_domain_valid_per_mic"])),
    )
    summary = {k: v for k, v in cand.items() if math.isfinite(v)}
    details["undefined"] = {k: _reason(k, ctx) for k, v in cand.items() if not math.isfinite(v)}
    details["coverage"] = dict(
        n_mics=real.shape[0],
        n_rotors=lab.shape[0],
        n_samples=real.shape[1],
        duration_s=real.shape[1] / sr,
        lsd_bins=details["spectrum"]["lsd_bins"],
        third_octave_bands=len(THIRD_OCTAVE_CENTRES_HZ),
        line_floor_line_bins_per_octave=lf["line_bins"],
        line_floor_floor_bins_per_octave=lf["floor_bins"],
        order_valid_cells_per_group=order_d["valid_cells_per_group"],
        order_visible_cells_per_group=order_d["visible_cells_per_group"],
        nominal80_mics_with_defined_sharpness=details["nominal80_psychoacoustics"][
            "mics_with_defined_sharpness"
        ],
        n_summary_defined=len(summary),
        n_summary_undefined=len(details["undefined"]),
    )
    return dict(summary=summary, details=_jsonable(details), metadata=_metadata(sr))
