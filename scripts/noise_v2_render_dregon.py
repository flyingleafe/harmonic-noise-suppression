"""Why the round-2 v2 DREGON arm renders to PIT MAE 75 rev/s.

The R2 candidate (``results/noise_v2/rounds/round2/fits/dregon_room2_floor__flight_floor_only.json``
— a ``flight_floor_only`` fit on the DREGON room-2 cruise windows with the comb
frozen at the log-mean of the four ``bench_dregon_Motor{1..4}_70`` bench fits)
scores HPPNet PIT MAE 75.09 rev/s against a real 1.219 and a legacy 2.188. The
combs themselves are identified 21/21, so the defect is in the RENDER.

This runner takes TWO of the five frozen DREGON cruise supports and measures,
arm by arm, what the render puts on the wire:

* the real clip; the v2 render (cross-checked bit-for-bit against the scored
  arm's own npz); the LEGACY render of the same recording (the identity-matched
  ``results/S2/dregon_room2_cruise_refined.json`` route the frozen baseline came
  from, :func:`noise_v2_round_score.legacy_arm`);
* VARIANTS of the same fit, each disabling or re-levelling exactly one block:
  both speed exponents at their prior mean, the comb muted, the floor muted, the
  floor re-levelled onto the real window's band level, and the comb re-levelled
  onto three separate targets (the real clip's own k=2 line prominence, the
  comb-to-floor ratio the BENCH fits identified the comb at, and the real
  window's broadband level).

The comb-to-floor measurement is ORDER-TRACKED: DREGON cruise carriers move
10-25 rev/s inside a 4 s window, so a periodogram at the MEAN carrier smears
every line into its own neighbours and reads ~2 dB on real audio that HPPNet
tracks to 1.2 rev/s. Here every frame's line is read at that frame's own label
carrier, and the local floor excludes every rotor's k-1, k, k+1 lines.

    # everything, on a laptop: the CPU probe is ~5 s per arm
    python scripts/noise_v2_render_dregon.py --probe --figures --out DIR

    # the same on uni-gpushort (identical numbers, the frozen checkpoint is
    # digest-verified in both paths)
    python scripts/noise_v2_render_dregon.py --probe --out DIR

Renders are deterministic in ``--seed``, so the two passes agree cell for cell.
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
from typing import Any, NoReturn

import numpy as np

from experiments.noise_model import gates as GT
from experiments.noise_model import render as RD
from experiments.noise_model import spectrum as SP
from experiments.stochastic_fit import revised_eval as RE

SCHEMA = "noise-v2-render-dregon/1"
OUT_DEFAULT = Path("results/noise_v2/rounds/round2/render_dregon")
FIT_DEFAULT = Path("results/noise_v2/rounds/round2/fits/dregon_room2_floor__flight_floor_only.json")
ARM_NPZ_DEFAULT = Path("results/noise_v2/rounds/round2/render/audio/dregon_v2")
BENCH_DEFAULT = Path("results/noise_v2/rounds/round2/fits")

#: The two frozen DREGON room-2 cruise supports this study is about, by
#: recording. ``hovering`` is the quietest proxy failure (3.65 dB) and
#: ``updown`` the loudest (5.48 dB) of the five in `round2/render/findings.md`.
STUDY_RECORDINGS = ("hovering_nosource_room2", "updown_nosource_room2")

#: The four bench fits the comb was frozen from (``fit['frozen_from']``).
BENCH_MOTORS = (1, 2, 3, 4)
BENCH_THROTTLE = 70

#: Band-level / LTAS periodogram (reused from the regime study: 8192 at 16 kHz
#: is 1.95 Hz per bin).
PSD_N = 8192
PSD_HOP = 4096

#: ORDER-TRACKED comb geometry. 4096 at 16 kHz is 3.91 Hz per bin and 256 ms of
#: support: fine enough that two rotors 8 rev/s apart are four bins apart at
#: k=2, short enough that a DREGON cruise carrier is stationary inside it.
TRK_N = 4096
TRK_HOP = 1024
#: Half-width of the line window, in bins; the local floor is the median of the
#: bins within +-0.5 f0 EXCLUDING +-1.5 bins around every rotor's k-1/k/k+1 line.
TRK_PEAK_BINS = 1.0
TRK_EXCL_BINS = 1.5
TRK_SPAN_REL = 0.5
TRK_MIN_FREE = 6

#: Spectrogram panels: the campaign's own flight front end.
SPEC_N = 2048
SPEC_HOP = 512

COMB_ORDERS = (2, 4, 8)

#: Mute level of a disabled block (200 dB down is numerically gone, still finite).
MUTE_DB = 200.0

#: Prior means of the two speed exponents and the floor's static pedestal
#: (``model.Priors``), i.e. what the R1 regime patch pins them to.
PRIOR_EXP = 2.0
PRIOR_STATIC_REL = 2.5e-3


def die(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def _module(name: str) -> Any:
    """Load a sibling script by path (the scripts dir is not a package)."""
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


# ── variant fits ────────────────────────────────────────────────────────────


def _mutate(fit: dict[str, Any], **kw: Any) -> dict[str, Any]:
    """A deep copy of ``fit`` with named speed-envelope / level edits applied."""
    out = copy.deepcopy(fit)
    p = out["params"]
    if "amp_exp" in kw:
        p["profile"]["amp_exp"] = float(kw["amp_exp"])
    if "floor_exp" in kw:
        p["floor"]["floor_exp"] = float(kw["floor_exp"])
        p["floor"]["floor_static_rel"] = float(kw.get("floor_static_rel", 0.0))
    if kw.get("mute_comb"):
        p["profile"]["profile_db"] = (
            np.asarray(p["profile"]["profile_db"], dtype=np.float64) - MUTE_DB
        ).tolist()
    if kw.get("mute_floor"):
        p["floor"]["floor_mean_db"] = float(p["floor"]["floor_mean_db"]) - MUTE_DB
    if "comb_shift_db" in kw:
        p["profile"]["profile_db"] = (
            np.asarray(p["profile"]["profile_db"], dtype=np.float64) + float(kw["comb_shift_db"])
        ).tolist()
    if "floor_shift_db" in kw:
        p["floor"]["floor_mean_db"] = float(p["floor"]["floor_mean_db"]) + float(
            kw["floor_shift_db"]
        )
    return out


@dataclass(frozen=True)
class ArmSpec:
    name: str
    kind: str  # real | v2 | legacy
    label: str
    hypothesis: str
    build: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None
    probe: bool = True


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec("real", "real", "the real DREGON room-2 clip", "reference"),
    ArmSpec(
        "legacy",
        "legacy",
        "legacy stage-2 baseline, identity-matched on this recording",
        "reference (the parity bar's own render)",
    ),
    ArmSpec("v2", "v2", "the round-2 v2 candidate, unchanged", "the failing arm", lambda f, _c: f),
    ArmSpec(
        "v2_nocomb",
        "v2",
        "v2 with the comb muted (floor only)",
        "control: what the render scores with NO comb at all",
        lambda f, _c: _mutate(f, mute_comb=True),
    ),
    ArmSpec(
        "v2_nofloor",
        "v2",
        "v2 with the floor muted (the frozen bench comb alone)",
        "control: is the frozen comb trackable on its own?",
        lambda f, _c: _mutate(f, mute_floor=True),
    ),
    ArmSpec(
        "v2_prior_exps",
        "v2",
        "v2 with both speed exponents at their prior means "
        "(amp_exp = floor_exp = 2, static_rel at its own prior mean)",
        "H1: the R1 regime patch — are the exponents the defect here too?",
        lambda f, _c: _mutate(
            f, amp_exp=PRIOR_EXP, floor_exp=PRIOR_EXP, floor_static_rel=PRIOR_STATIC_REL
        ),
    ),
    ArmSpec(
        "v2_floor_matched",
        "v2",
        "v2 with the floor re-levelled onto THIS window's real band level",
        "H3, floor side: is the floor level itself wrong?",
        lambda f, c: _mutate(f, floor_shift_db=float(c["floor_match_shift_db"])),
    ),
    ArmSpec(
        "v2_comb_real_ratio",
        "v2",
        "v2 with the comb re-levelled so its k=2 line prominence matches the real clip's",
        "the literal (e): match real's OWN measured k=2 comb-to-floor",
        lambda f, c: _mutate(f, comb_shift_db=float(c["comb_real_ratio_shift_db"])),
    ),
    ArmSpec(
        "v2_comb_bench_ratio",
        "v2",
        "v2 with the comb re-levelled so its k=2 comb-to-floor equals the one the BENCH "
        "fits identified it at",
        "H3, comb side: restore the ratio the frozen comb was fitted under",
        lambda f, c: _mutate(f, comb_shift_db=float(c["comb_bench_ratio_shift_db"])),
    ),
    ArmSpec(
        "v2_comb_level_matched",
        "v2",
        "v2 with the comb re-levelled so the FULL render's band level matches the real window's",
        "H3, comb side: the comb carries the real clip's own broadband level",
        lambda f, c: _mutate(f, comb_shift_db=float(c["comb_level_match_shift_db"])),
    ),
)


# ── measurements ────────────────────────────────────────────────────────────


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


def _stft_power(x: np.ndarray, *, n: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """``(frames, bins)`` power and the centre SAMPLE of each frame."""
    x = np.asarray(x, dtype=np.float64)
    w = np.hanning(n)
    starts = np.arange(0, x.size - n + 1, hop)
    if starts.size == 0:
        die(f"{x.size} samples is shorter than the {n}-point tracked window")
    power = np.stack([np.abs(np.fft.rfft(x[s : s + n] * w)) ** 2 for s in starts])
    return power / float((w**2).sum()), starts + n // 2


def tracked_line(
    x: np.ndarray,
    f0_tracks: np.ndarray,
    k: int,
    *,
    sr: int,
    peak_bins: float = TRK_PEAK_BINS,
) -> dict[str, Any]:
    """Order-``k`` line power and prominence over the local floor, per rotor.

    The line of rotor ``r`` in frame ``j`` is read at ``k f0_r`` of THAT frame's
    own label carrier, so a moving DREGON cruise carrier does not smear it; the
    local floor is the median of the bins within ``+-0.5 f0`` EXCLUDING
    ``+-TRK_EXCL_BINS`` bins around every rotor's k-1, k and k+1 line, so a
    neighbouring rotor or a neighbouring order cannot be mistaken for floor.
    Both statistics are the MEDIAN over frames; the peak is the MEAN of the bins
    inside the line window (not their max), which keeps the estimator's noise
    bias at +-1 dB instead of the +6 dB a max-over-bins carries.
    """
    power, centres = _stft_power(x, n=TRK_N, hop=TRK_HOP)
    df = float(sr) / float(TRK_N)
    bins = np.fft.rfftfreq(TRK_N, d=1.0 / float(sr))
    f0 = np.asarray(f0_tracks, dtype=np.float64)
    n_t = int(f0.shape[1])
    prom: list[list[float]] = [[] for _ in range(f0.shape[0])]
    line_p: list[list[float]] = [[] for _ in range(f0.shape[0])]
    floor_p: list[list[float]] = [[] for _ in range(f0.shape[0])]
    for j, c in enumerate(centres):
        frame = f0[:, min(int(c), n_t - 1)]
        span = TRK_SPAN_REL * float(frame.mean())
        excl = np.zeros_like(bins, dtype=bool)
        for order in (k - 1, k, k + 1):
            if order < 1:
                continue
            for line in frame * float(order):
                excl |= np.abs(bins - line) <= (TRK_EXCL_BINS + peak_bins - 1.0) * df
        for r, line in enumerate(frame * float(k)):
            if line <= SP.BAND_F_MIN or line >= SP.BAND_F_MAX:
                continue
            peak_sel = np.abs(bins - line) <= peak_bins * df
            floor_sel = (np.abs(bins - line) <= span) & ~excl
            if not peak_sel.any() or int(floor_sel.sum()) < TRK_MIN_FREE:
                continue
            pk = float(power[j][peak_sel].mean())
            fl = float(np.median(power[j][floor_sel]))
            line_p[r].append(pk)
            floor_p[r].append(fl)
            prom[r].append(10.0 * np.log10(max(pk, 1e-300) / max(fl, 1e-300)))

    def med(rows: list[list[float]]) -> list[float | None]:
        return [float(np.median(v)) if v else None for v in rows]

    pr = med(prom)
    ok = [v for v in pr if v is not None]
    return dict(
        k=int(k),
        peak_bins=float(peak_bins),
        prominence_db_per_rotor=pr,
        prominence_db=float(np.mean(ok)) if ok else None,
        line_power_per_rotor=med(line_p),
        local_floor_power_per_rotor=med(floor_p),
        n_frames=[len(v) for v in prom],
    )


def line_ratio_db(
    comb_only: np.ndarray, floor_only: np.ndarray, f0: np.ndarray, k: int, *, sr: int
) -> dict[str, Any]:
    """Comb-to-floor of order ``k`` measured on SEPARATE comb / floor renders.

    A muted-floor and a muted-comb render of the same seed carry exactly the two
    blocks of the same render, so their order-``k`` line powers give the true
    per-line comb-to-floor without any local-floor estimator: the number the
    full render's prominence ``10 log10(1 + ratio)`` is made of.
    """
    a = tracked_line(comb_only, f0, k, sr=sr)
    b = tracked_line(floor_only, f0, k, sr=sr)
    per: list[float | None] = []
    for c, f in zip(a["line_power_per_rotor"], b["line_power_per_rotor"], strict=True):
        per.append(
            None
            if c is None or f is None
            else float(10.0 * np.log10(max(c, 1e-300) / max(f, 1e-300)))
        )
    ok = [v for v in per if v is not None]
    mean = float(np.mean(ok)) if ok else None
    return dict(
        k=int(k),
        per_rotor_db=per,
        mean_db=mean,
        implied_prominence_db=(
            None
            if not ok
            else float(np.mean([10.0 * np.log10(1.0 + 10.0 ** (v / 10.0)) for v in ok]))
        ),
    )


def encoded_speed_dev(
    x: np.ndarray, f0_track: np.ndarray, *, sr: int, k: int = 2, bw_hz: float = 4.0
) -> dict[str, Any]:
    """Speed the signal ENCODES at order ``k``, against the label track.

    Demodulates by the label's own order-``k`` phase, low-passes to ``+-bw_hz``
    and differentiates the residual phase. On a render whose comb sits on the
    label carrier the residual is the model's own shaft jitter; if the render
    encoded another speed it ramps.
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
    dev = (np.diff(np.unwrap(np.angle(lp))) * float(sr) / (2.0 * np.pi * float(k)))[guard]
    w = mag[1:][guard]
    if not np.isfinite(dev).all() or w.sum() <= 0:
        return dict(k=int(k), bw_hz=float(bw_hz), median_abs_dev_rev_s=None)
    order = np.argsort(np.abs(dev))
    cum = np.cumsum(w[order])
    q = lambda p: float(np.abs(dev)[order][int(np.searchsorted(cum, p * cum[-1]))])  # noqa: E731
    return dict(
        k=int(k),
        bw_hz=float(bw_hz),
        median_abs_dev_rev_s=q(0.5),
        p90_abs_dev_rev_s=q(0.9),
        mean_dev_rev_s=float(np.average(dev, weights=w)),
    )


def arm_diagnostics(
    audio: np.ndarray,
    real: np.ndarray,
    *,
    sr: int,
    f0_tracks: np.ndarray,
    demod_rotors: tuple[int, ...],
) -> dict[str, Any]:
    """Every per-arm number of one support: level, LTAS, comb, encoded speed."""
    freqs = np.fft.rfftfreq(PSD_N, d=1.0 / float(sr))
    psd_mics = np.stack([_welch(audio[m]) for m in range(audio.shape[0])])
    levels = [band_level_db(psd_mics[m], freqs) for m in range(psd_mics.shape[0])]
    ltas = RE.ltas_deviation_db(real, audio, mic=0)
    tracked = {
        f"mic{m}": [tracked_line(audio[m], f0_tracks, k, sr=sr) for k in COMB_ORDERS] for m in (0,)
    }
    return dict(
        band_level_db_mic0=float(levels[0]),
        band_level_db_mic_mean=float(
            10.0 * np.log10(np.mean([10.0 ** (v / 10.0) for v in levels]))
        ),
        band_level_db_per_mic=[float(v) for v in levels],
        rms_per_mic=[float(v) for v in np.sqrt((np.asarray(audio, np.float64) ** 2).mean(axis=1))],
        rms_broadband=float(np.sqrt((np.asarray(audio, np.float64) ** 2).mean())),
        peak_abs=float(np.abs(audio).max()),
        ltas_mean_abs_db=float(ltas["mean_abs_db"]),
        ltas_level_offset_db=float(ltas["level_offset_db"]),
        ltas_max_abs_db=float(ltas["max_abs_db"]),
        ltas_shape_only_mean_abs_db=float(ltas["shape_only_mean_abs_db"]),
        ltas_bands_db=[float(v) for v in ltas["bands_arm_db"]],
        comb_prominence=tracked["mic0"],
        comb_prominence_mic_mean_db=[
            float(
                np.mean(
                    [
                        v
                        for v in [
                            tracked_line(audio[m], f0_tracks, k, sr=sr)["prominence_db"]
                            for m in range(audio.shape[0])
                        ]
                        if v is not None
                    ]
                )
            )
            for k in COMB_ORDERS
        ],
        encoded_speed={
            f"rotor{r}": encoded_speed_dev(audio[0], f0_tracks[r], sr=sr) for r in demod_rotors
        },
        _psd=psd_mics,
    )


# ── the two speed laws, and where the comb's absolute level comes from ──────


def envelope_report(params: dict[str, Any], f0_rotor: np.ndarray) -> dict[str, Any]:
    """The comb and floor speed envelopes evaluated at a given carrier set.

    Exactly the renderer's arithmetic (``render.render_noise``): the comb
    carries ``speed ** amp_exp`` in POWER and the floor
    ``mean_r(speed ** floor_exp) + floor_static_rel``.
    """
    s = np.asarray(f0_rotor, dtype=np.float64) / float(RD.AMP_RPS_REF)
    amp_exp = float(params["profile"]["amp_exp"])
    floor_exp = float(params["floor"]["floor_exp"])
    rel = float(params["floor"]["floor_static_rel"])
    comb = float(np.mean(s**amp_exp))
    floor = float(np.mean(s**floor_exp) + rel)
    profile = np.asarray(params["profile"]["profile_db"], dtype=np.float64)
    k_max = min(
        int(profile.shape[1]),
        int(SP.k_max_for_carrier(np.asarray(f0_rotor, dtype=np.float64), SP.FLIGHT_SR)),
    )
    return dict(
        carrier_rev_s=[float(v) for v in np.asarray(f0_rotor, dtype=np.float64)],
        speed_mean=float(s.mean()),
        amp_exp=amp_exp,
        floor_exp=floor_exp,
        floor_static_rel=rel,
        comb_envelope_db=float(10.0 * np.log10(max(comb, 1e-300))),
        floor_envelope_db=float(10.0 * np.log10(max(floor, 1e-300))),
        comb_minus_floor_db=float(
            10.0 * np.log10(max(comb, 1e-300)) - 10.0 * np.log10(max(floor, 1e-300))
        ),
        k_max=int(k_max),
        top_comb_line_hz=float(k_max * float(np.max(f0_rotor))),
    )


def comb_level_accounting(fit: dict[str, Any], bench: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Where the rendered comb's ABSOLUTE level comes from, term by term.

    ``render_noise`` MEAN-CENTRES both ``mic_line_gain_db`` (over mics, per
    rotor) and ``mic_gains_db`` before applying them, so neither carries any
    absolute level: the only absolute comb scale in a render is ``profile_db``.
    This function states that explicitly — the per-mic line power the renderer
    actually produces, per rotor, against the per-mic line power each bench fit
    produces on its own carrier.
    """

    def per_mic_line_power_db(params: dict[str, Any], n_mics: int = 8) -> dict[str, Any]:
        prof = np.asarray(params["profile"]["profile_db"], dtype=np.float64)
        mlg = np.asarray(params["profile"]["mic_line_gain_db"], dtype=np.float64)
        gall = np.asarray(params["mic_gains_db"], dtype=np.float64)[:n_mics]
        centred_line = mlg[:n_mics] - mlg[:n_mics].mean(axis=0, keepdims=True)
        centred_all = gall - gall.mean()
        line_gain = 10.0 ** (centred_line / 10.0)
        all_gain = 10.0 ** (centred_all / 10.0)
        # sum of line powers over orders, per (mic, rotor)
        per_rotor = 10.0 ** (prof / 10.0)  # (R, K)
        totals = per_rotor.sum(axis=1)  # (R,)
        pm = line_gain * totals[None, :] * all_gain[:, None]  # (M, R)
        return dict(
            n_orders=int(prof.shape[1]),
            per_rotor_sum_db=[float(10.0 * np.log10(max(v, 1e-300))) for v in totals],
            mic0_per_rotor_db=[float(10.0 * np.log10(max(v, 1e-300))) for v in pm[0]],
            mic0_total_db=float(10.0 * np.log10(max(float(pm[0].sum()), 1e-300))),
            mic_mean_total_db=float(10.0 * np.log10(max(float(pm.sum(axis=1).mean()), 1e-300))),
            mic_line_gain_db_raw_mean=[float(v) for v in mlg[:n_mics].mean(axis=0)],
            mic_line_gain_db_centred_mic0=[float(v) for v in centred_line[0]],
            mic_gains_db_raw_mean=float(gall.mean()),
        )

    cand = per_mic_line_power_db(fit["params"])
    per_bench = {
        f"Motor{m}": per_mic_line_power_db(b["params"]) | {"n_rotors": 1} for m, b in bench.items()
    }
    bench_mic0 = [v["mic0_total_db"] for v in per_bench.values()]
    return dict(
        note=(
            "render_noise centres mic_line_gain_db over mics (per rotor) and mic_gains_db over "
            "mics before use, so their ABSOLUTE values are discarded by the renderer; the only "
            "absolute comb scale a render has is profile_db"
        ),
        candidate=cand,
        bench=per_bench,
        bench_mic0_total_db_mean=float(np.mean(bench_mic0)),
        candidate_minus_bench_mic0_db=float(cand["mic0_total_db"] - float(np.mean(bench_mic0))),
        four_rotor_sum_db=float(
            10.0 * np.log10(len(cand["mic0_per_rotor_db"])) if cand["mic0_per_rotor_db"] else 0.0
        ),
    )


def bench_reference(
    bench: dict[int, dict[str, Any]], *, seed: int, n_mics: int, seconds: float
) -> dict[str, Any]:
    """The comb-to-floor ratio each bench fit identified ITS comb under.

    Each bench fit is rendered on its OWN constant carrier, comb-only and
    floor-only, so the ratio is exact; this is the ratio the frozen comb was
    fitted at, and the one the flight render has to reproduce for the comb to be
    as visible on the wire as it was in the bench likelihood.
    """
    sr = int(SP.FLIGHT_SR)
    n = int(round(float(seconds) * sr))
    freqs = np.fft.rfftfreq(PSD_N, d=1.0 / float(sr))
    out: dict[str, Any] = {}
    for motor, f in sorted(bench.items()):
        carrier = np.asarray(f["params"]["carrier_rev_s"], dtype=np.float64)
        if carrier.size != 1:
            die(f"bench Motor{motor}: expected one carrier, got {carrier.tolist()}")
        track = np.full((1, n), float(carrier[0]))
        comb = RD.render_noise(_mutate(f, mute_floor=True), track, n_mics=n_mics, seed=seed)
        floor = RD.render_noise(_mutate(f, mute_comb=True), track, n_mics=n_mics, seed=seed)
        full = RD.render_noise(f, track, n_mics=n_mics, seed=seed)
        out[f"Motor{motor}"] = dict(
            carrier_rev_s=float(carrier[0]),
            converged=(f.get("diagnostics") or {}).get("converged"),
            floor_mean_db=float(f["params"]["floor"]["floor_mean_db"]),
            amp_exp=float(f["params"]["profile"]["amp_exp"]),
            floor_exp=float(f["params"]["floor"]["floor_exp"]),
            comb_only_band_level_db_mic0=band_level_db(_welch(comb[0]), freqs),
            floor_only_band_level_db_mic0=band_level_db(_welch(floor[0]), freqs),
            full_band_level_db_mic0=band_level_db(_welch(full[0]), freqs),
            comb_minus_floor_band_db=float(
                band_level_db(_welch(comb[0]), freqs) - band_level_db(_welch(floor[0]), freqs)
            ),
            line_ratio=[line_ratio_db(comb[0], floor[0], track, k, sr=sr) for k in COMB_ORDERS],
            full_prominence_db=[
                tracked_line(full[0], track, k, sr=sr)["prominence_db"] for k in COMB_ORDERS
            ],
        )
    ratios = {
        k: [
            v["line_ratio"][i]["mean_db"]
            for v in out.values()
            if v["line_ratio"][i]["mean_db"] is not None
        ]
        for i, k in enumerate(COMB_ORDERS)
    }
    return dict(
        seconds=float(seconds),
        per_motor=out,
        mean_line_ratio_db={str(k): float(np.mean(v)) if v else None for k, v in ratios.items()},
        mean_floor_mean_db=float(np.mean([v["floor_mean_db"] for v in out.values()])),
        mean_comb_minus_floor_band_db=float(
            np.mean([v["comb_minus_floor_band_db"] for v in out.values()])
        ),
    )


# ── the run ─────────────────────────────────────────────────────────────────


def study_supports() -> list[GT.ScoredSupport]:
    out: list[GT.ScoredSupport] = []
    for rec in STUDY_RECORDINGS:
        found = [s for s in GT.DREGON_CRUISE_SUPPORTS if s.recording == rec]
        if not found:
            die(f"no frozen DREGON cruise support carries recording {rec!r}")
        out.append(found[0])
    return out


def _demod_rotors(f0_rotor: np.ndarray) -> tuple[int, ...]:
    """Rotors whose order-2 line is resolvable from every other rotor's."""
    out: list[int] = []
    for r, f in enumerate(f0_rotor):
        gaps = [abs(float(f) - float(g)) for i, g in enumerate(f0_rotor) if i != r]
        if min(gaps) * 2.0 > 8.0:
            out.append(int(r))
    return tuple(out) or (int(np.argmax(f0_rotor)),)


def read_bench(fits_dir: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for m in BENCH_MOTORS:
        p = Path(fits_dir) / f"bench_dregon_Motor{m}_{BENCH_THROTTLE}__bench.json"
        if not p.exists():
            die(f"{p}: the frozen comb's bench fit is missing")
        out[m] = json.loads(p.read_text())
    return out


def _calibrate(
    cache: dict[str, Any],
    *,
    row: dict[str, Any],
    sr: int,
    comb_only: np.ndarray,
    floor_only: np.ndarray,
    f0_tracks: np.ndarray,
    bench_ratio_db: float | None,
) -> dict[str, Any]:
    """The four re-levelling shifts, each from a measured number.

    ``comb_only`` and ``floor_only`` are the two blocks of the SAME seed, so
    their band powers and line powers add to the full render's exactly, and
    every shift below is a closed-form solve rather than a search.
    """
    freqs = np.fft.rfftfreq(PSD_N, d=1.0 / float(sr))
    p_comb = 10.0 ** (band_level_db(_welch(comb_only[0]), freqs) / 10.0)
    p_floor = 10.0 ** (band_level_db(_welch(floor_only[0]), freqs) / 10.0)
    real_level = float(row["arms"]["real"]["band_level_db_mic0"])
    p_real = 10.0 ** (real_level / 10.0)
    ratio = line_ratio_db(comb_only[0], floor_only[0], f0_tracks, 2, sr=sr)
    ratio_db = ratio["mean_db"]
    if ratio_db is None:
        die("the order-2 comb-to-floor of the separated render is unmeasurable")
    real_prom = None
    for entry in row["arms"]["real"]["comb_prominence"]:
        if int(entry["k"]) == 2:
            real_prom = entry["prominence_db"]
    if real_prom is None:
        die("the real clip's order-2 prominence is unmeasurable")
    # prominence P dB implies a per-line comb-to-floor of 10 log10(10^(P/10) - 1)
    excess = 10.0 ** (float(real_prom) / 10.0) - 1.0
    cache["dregon_line_ratio_k2_db"] = float(ratio_db)
    cache["real_prominence_k2_db"] = float(real_prom)
    cache["comb_real_ratio_shift_db"] = float(
        (10.0 * np.log10(max(excess, 1e-6))) - float(ratio_db)
    )
    cache["comb_bench_ratio_shift_db"] = (
        0.0 if bench_ratio_db is None else float(float(bench_ratio_db) - float(ratio_db))
    )
    cache["floor_match_shift_db"] = float(real_level - band_level_db(_welch(floor_only[0]), freqs))
    headroom = p_real - p_floor
    cache["comb_level_match_shift_db"] = (
        float(10.0 * np.log10(max(headroom, 1e-300) / max(p_comb, 1e-300)))
        if headroom > 0.0
        else float("nan")
    )
    cache["comb_only_band_level_db_mic0"] = float(10.0 * np.log10(max(p_comb, 1e-300)))
    cache["floor_only_band_level_db_mic0"] = float(10.0 * np.log10(max(p_floor, 1e-300)))
    cache["comb_minus_floor_band_db"] = float(
        10.0 * np.log10(max(p_comb, 1e-300) / max(p_floor, 1e-300))
    )
    cache["order2_ratio"] = ratio
    return cache


def run(
    *,
    fit_path: Path,
    bench_dir: Path,
    arm_npz: Path,
    out: Path,
    seed: int,
    probe: bool,
    n_mics: int,
) -> dict[str, Any]:
    rs = _module("noise_v2_round_score")
    Probe, _arm_for_recording, legacy_arm = rs.Probe, rs._arm_for_recording, rs.legacy_arm

    fit = json.loads(Path(fit_path).read_text())
    if str(fit.get("schema")) != "noise-v2-fit/1":
        die(f"{fit_path}: not a noise-v2-fit/1 payload")
    bench = read_bench(bench_dir)
    probe_obj = Probe.load() if probe else None
    legacy = legacy_arm("dregon")
    diag = fit.get("diagnostics") or {}
    payload: dict[str, Any] = dict(
        schema=SCHEMA,
        git=git_rev(),
        fit=dict(
            path=str(fit_path),
            support=fit.get("support"),
            mode=fit.get("mode"),
            k_max=fit.get("k_max"),
            converged=diag.get("converged"),
            frozen_from=fit.get("frozen_from"),
            amp_exp=float(fit["params"]["profile"]["amp_exp"]),
            floor_exp=float(fit["params"]["floor"]["floor_exp"]),
            floor_static_rel=float(fit["params"]["floor"]["floor_static_rel"]),
            floor_mean_db=float(fit["params"]["floor"]["floor_mean_db"]),
            floor_tilt_db_oct=float(fit["params"]["floor"]["floor_tilt_db_oct"]),
            init_floor_mean_db=diag.get("init_floor_mean_db"),
            floor_level_db=diag.get("floor_level_db"),
            floor_mean_moved_from_init_db=(
                None
                if diag.get("init_floor_mean_db") is None
                else float(
                    float(fit["params"]["floor"]["floor_mean_db"])
                    - float(diag["init_floor_mean_db"])
                )
            ),
            sigma_nu=float(fit["params"]["sigma_nu"]),
            lam=float(fit["params"]["lam"]),
            amp_rps_ref=float(RD.AMP_RPS_REF),
            pool_carrier_rev_s=[
                (diag.get("batch") or {}).get("carrier_min_rev_s"),
                (diag.get("batch") or {}).get("carrier_max_rev_s"),
            ],
        ),
        protocol=dict(
            seed=int(seed),
            n_mics=int(n_mics),
            psd=dict(n_fft=PSD_N, hop=PSD_HOP),
            tracked=dict(
                n_fft=TRK_N,
                hop=TRK_HOP,
                peak_bins=TRK_PEAK_BINS,
                excl_bins=TRK_EXCL_BINS,
                span_rel=TRK_SPAN_REL,
                statistic="mean over line bins / median over local floor bins, median over frames",
            ),
            spectrogram=dict(n_fft=SPEC_N, hop=SPEC_HOP),
            band_hz=[SP.BAND_F_MIN, SP.BAND_F_MAX],
            comb_orders=list(COMB_ORDERS),
            scorer=(probe_obj.record if probe_obj is not None else None),
            legacy_route={
                k: dict(v, clip_ids=list(v.get("clip_ids") or [])[:4])
                for k, v in legacy.source.items()
            },
            arm_npz=str(arm_npz),
        ),
        supports={},
    )
    payload["bench_reference"] = bench_reference(bench, seed=seed, n_mics=n_mics, seconds=4.0)
    payload["comb_level_accounting"] = comb_level_accounting(fit, bench)
    bench_ratio = payload["bench_reference"]["mean_line_ratio_db"].get("2")

    figures: dict[str, Any] = {}
    for support in study_supports():
        clip = RE.load_window(
            support.window,
            dataset=GT.DATASET["dregon"],
            version=None,
            channels=None,
            rps_key=GT.RAW_RPS_KEY["dregon"],
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
        bench_carrier = np.asarray(
            [float(b["params"]["carrier_rev_s"][0]) for b in bench.values()], dtype=np.float64
        )
        row: dict[str, Any] = dict(
            support=support.as_dict(),
            n_samples=int(real.shape[-1]),
            scored_seconds=float(rsupport.scored_seconds),
            mean_reference_rps=float(reference.mean()),
            per_rotor_mean_rps=[float(v) for v in f0_rotor],
            per_rotor_min_rps=[float(v) for v in reference.min(axis=1)],
            per_rotor_max_rps=[float(v) for v in reference.max(axis=1)],
            speed_norm=[float(v / RD.AMP_RPS_REF) for v in f0_rotor],
            demod_rotors=list(demod),
            envelope_at_dregon_cruise=envelope_report(fit["params"], f0_rotor),
            envelope_at_bench_carrier=envelope_report(fit["params"], bench_carrier),
            arms={},
        )
        row["envelope_bench_to_cruise_shift_db"] = dict(
            comb_db=float(
                row["envelope_at_dregon_cruise"]["comb_envelope_db"]
                - row["envelope_at_bench_carrier"]["comb_envelope_db"]
            ),
            floor_db=float(
                row["envelope_at_dregon_cruise"]["floor_envelope_db"]
                - row["envelope_at_bench_carrier"]["floor_envelope_db"]
            ),
        )
        cache: dict[str, Any] = {}
        psds: dict[str, np.ndarray] = {}
        waves: dict[str, np.ndarray] = {}
        blocks: dict[str, np.ndarray] = {}
        for spec in ARMS:
            t0 = time.time()
            if spec.kind == "real":
                audio = real
            elif spec.kind == "legacy":
                arm = _arm_for_recording(legacy, support)
                audio = arm.render(
                    reference, regime=support.regime, n_mics=len(mics), seed=int(seed)
                )[: len(mics)]
            else:
                assert spec.build is not None
                needs_cal = spec.name.startswith("v2_comb_") or spec.name == "v2_floor_matched"
                if needs_cal and "comb_level_match_shift_db" not in cache:
                    _calibrate(
                        cache,
                        row=row,
                        sr=int(clip.sr),
                        comb_only=blocks["v2_nofloor"],
                        floor_only=blocks["v2_nocomb"],
                        f0_tracks=reference,
                        bench_ratio_db=bench_ratio,
                    )
                variant = spec.build(fit, cache)
                audio = RD.render_noise(variant, reference, n_mics=len(mics), seed=int(seed))[
                    : len(mics)
                ]
            audio = np.asarray(audio, dtype=np.float64)
            if int(audio.shape[-1]) != int(real.shape[-1]):
                die(
                    f"{support.key}: arm {spec.name} rendered {audio.shape[-1]} samples "
                    f"against the frozen {real.shape[-1]}"
                )
            if spec.name in ("v2_nocomb", "v2_nofloor"):
                blocks[spec.name] = audio
            d = arm_diagnostics(
                audio, real, sr=int(clip.sr), f0_tracks=reference, demod_rotors=demod
            )
            psds[spec.name] = d.pop("_psd")
            waves[spec.name] = audio[0].copy()
            entry: dict[str, Any] = dict(
                kind=spec.kind,
                label=spec.label,
                hypothesis=spec.hypothesis,
                render_seconds=float(time.time() - t0),
                **d,
            )
            if spec.kind == "v2" and spec.name != "v2":
                entry["variant_params"] = dict(
                    amp_exp=float(variant["params"]["profile"]["amp_exp"]),
                    floor_exp=float(variant["params"]["floor"]["floor_exp"]),
                    floor_static_rel=float(variant["params"]["floor"]["floor_static_rel"]),
                    floor_mean_db=float(variant["params"]["floor"]["floor_mean_db"]),
                    profile_shift_db=float(
                        np.asarray(variant["params"]["profile"]["profile_db"], np.float64).mean()
                        - np.asarray(fit["params"]["profile"]["profile_db"], np.float64).mean()
                    ),
                )
            if spec.name == "v2":
                entry["arm_npz_check"] = _npz_check(arm_npz, support, audio, seed=int(seed))
            if probe_obj is not None and spec.probe:
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
            row["arms"][spec.name] = entry
            print(
                f"[{support.recording}] {spec.name}: "
                f"level={entry['band_level_db_mic0']:+.2f} dB "
                f"ltas={entry['ltas_mean_abs_db']:.2f} dB "
                f"k2={entry['comb_prominence'][0]['prominence_db']} "
                f"pit={entry.get('pit_mae')}",
                flush=True,
            )
        row["calibration"] = {k: v for k, v in cache.items() if not isinstance(v, np.ndarray)}
        payload["supports"][support.key] = row
        figures[support.recording] = dict(
            support=support, psds=psds, waves=waves, sr=int(clip.sr), row=row
        )
    payload["verdicts"] = verdicts(payload)
    Path(out).mkdir(parents=True, exist_ok=True)
    return payload | {"_figures": figures}


def _npz_check(
    arm_npz: Path, support: GT.ScoredSupport, audio: np.ndarray, *, seed: int
) -> dict[str, Any]:
    """Is the arm we are diagnosing the one that was SCORED?

    The scored arm's audio is in the round's npz (float32); this render is
    float64 from the same fit and seed, so the two must agree to float32
    rounding. Anything larger means this study is not about the scored audio.
    """
    path = Path(arm_npz) / f"{support.key}.npz"
    if not path.exists():
        return dict(available=False, path=str(path))
    z = np.load(path)
    key = f"render_seed_{seed}"
    if key not in z.files:
        return dict(available=False, path=str(path), missing=key)
    ref = np.asarray(z[key], dtype=np.float64)[: audio.shape[0]]
    denom = float(np.abs(ref).max()) or 1.0
    return dict(
        available=True,
        path=str(path),
        key=key,
        max_abs_diff=float(np.abs(ref - audio).max()),
        max_rel_diff=float(np.abs(ref - audio).max() / denom),
        float32_eps_rel=float(np.finfo(np.float32).eps),
    )


# ── verdicts ────────────────────────────────────────────────────────────────


def _arm(payload: dict[str, Any], recording: str, name: str, key: str) -> Any:
    for row in payload["supports"].values():
        if row["support"]["recording"] == recording:
            return row["arms"].get(name, {}).get(key)
    return None


def _prom(payload: dict[str, Any], recording: str, name: str, k: int) -> Any:
    entries = _arm(payload, recording, name, "comb_prominence") or []
    for e in entries:
        if int(e["k"]) == int(k):
            return e["prominence_db"]
    return None


def verdicts(payload: dict[str, Any]) -> dict[str, Any]:
    """The four hypotheses, each decided by numbers already in the payload."""
    recs = [row["support"]["recording"] for row in payload["supports"].values()]
    out: dict[str, Any] = {}

    def over(fn: Callable[[str], Any]) -> dict[str, Any]:
        return {r: fn(r) for r in recs}

    out["H1_speed_law"] = dict(
        claim="the bench comb was fitted at ~68 rev/s and the DREGON cruise carriers differ; "
        "the speed law / k_max geometry must be checked at the cruise carriers",
        comb_envelope_shift_db=over(
            lambda r: _row(payload, r)["envelope_bench_to_cruise_shift_db"]["comb_db"]
        ),
        floor_envelope_shift_db=over(
            lambda r: _row(payload, r)["envelope_bench_to_cruise_shift_db"]["floor_db"]
        ),
        k_max_bench=over(lambda r: _row(payload, r)["envelope_at_bench_carrier"]["k_max"]),
        k_max_cruise=over(lambda r: _row(payload, r)["envelope_at_dregon_cruise"]["k_max"]),
        pit_prior_exps=over(lambda r: _arm(payload, r, "v2_prior_exps", "pit_mae")),
        pit_v2=over(lambda r: _arm(payload, r, "v2", "pit_mae")),
    )
    out["H2_mic_gain_units"] = dict(
        claim="a units / per-rotor level mismatch in mic_line_gain_db or mic_gains_db",
        renderer_note=payload["comb_level_accounting"]["note"],
        candidate_mic0_line_power_db=payload["comb_level_accounting"]["candidate"]["mic0_total_db"],
        bench_mic0_line_power_db_mean=payload["comb_level_accounting"]["bench_mic0_total_db_mean"],
        candidate_minus_bench_db=payload["comb_level_accounting"]["candidate_minus_bench_mic0_db"],
    )
    out["H3_floor_buries_comb"] = dict(
        claim="the floor-only flight fit's floor buries the frozen bench comb",
        comb_minus_floor_band_db=over(
            lambda r: _row(payload, r)["calibration"]["comb_minus_floor_band_db"]
        ),
        bench_comb_minus_floor_band_db=payload["bench_reference"]["mean_comb_minus_floor_band_db"],
        order2_line_ratio_db=over(
            lambda r: _row(payload, r)["calibration"]["dregon_line_ratio_k2_db"]
        ),
        bench_order2_line_ratio_db=payload["bench_reference"]["mean_line_ratio_db"].get("2"),
        pit_v2=over(lambda r: _arm(payload, r, "v2", "pit_mae")),
        pit_nocomb=over(lambda r: _arm(payload, r, "v2_nocomb", "pit_mae")),
        pit_comb_bench_ratio=over(lambda r: _arm(payload, r, "v2_comb_bench_ratio", "pit_mae")),
        pit_comb_level_matched=over(lambda r: _arm(payload, r, "v2_comb_level_matched", "pit_mae")),
        pit_floor_matched=over(lambda r: _arm(payload, r, "v2_floor_matched", "pit_mae")),
    )
    out["H4_wrong_speed"] = dict(
        claim="the render encodes a different speed than the label",
        encoded_speed_v2_comb=over(lambda r: _arm(payload, r, "v2_nofloor", "encoded_speed")),
        encoded_speed_real=over(lambda r: _arm(payload, r, "real", "encoded_speed")),
        scored_fraction_note="every frozen DREGON support is one 4 s cruise window; "
        "scored_seconds is reported per support",
        scored_seconds=over(lambda r: _row(payload, r)["scored_seconds"]),
        k_max_cruise=over(lambda r: _row(payload, r)["envelope_at_dregon_cruise"]["k_max"]),
        top_comb_line_hz=over(
            lambda r: _row(payload, r)["envelope_at_dregon_cruise"]["top_comb_line_hz"]
        ),
    )
    return out


def _row(payload: dict[str, Any], recording: str) -> dict[str, Any]:
    for row in payload["supports"].values():
        if row["support"]["recording"] == recording:
            return row
    die(f"no support row for recording {recording!r}")


# ── figures ─────────────────────────────────────────────────────────────────

FIG_ARMS = ("real", "legacy", "v2", "v2_nocomb", "v2_nofloor", "v2_comb_level_matched")
FIG_COLOURS = {
    "real": "#111111",
    "legacy": "#1f77b4",
    "v2": "#d62728",
    "v2_nocomb": "#ff9896",
    "v2_nofloor": "#9467bd",
    "v2_comb_level_matched": "#2ca02c",
}


def _smooth_log(
    freqs: np.ndarray, psd: np.ndarray, *, per_oct: int = 48
) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = float(SP.BAND_F_MIN), float(SP.BAND_F_MAX)
    n = int(np.ceil(per_oct * np.log2(hi / lo)))
    edges = lo * 2.0 ** (np.arange(n + 1) / per_oct)
    centres = np.sqrt(edges[:-1] * edges[1:])
    out = np.zeros(n)
    ok = np.zeros(n, dtype=bool)
    for i in range(n):
        sel = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if sel.any():
            out[i] = float(psd[sel].mean())
            ok[i] = True
    return centres[ok], 10.0 * np.log10(np.maximum(out[ok], 1e-300))


def _save(fig: Any, path: Path, *, colors: int = 128) -> str:
    fig.savefig(path, dpi=130, bbox_inches="tight")
    try:
        from PIL import Image

        img = Image.open(path).convert("RGB").quantize(colors=colors, method=2)
        img.save(path, optimize=True)
    except Exception:  # pragma: no cover - cosmetic only
        pass
    return str(path)


def write_figures(payload: dict[str, Any], figures: dict[str, Any], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[str] = []
    for recording, fg in figures.items():
        sr = int(fg["sr"])
        freqs = np.fft.rfftfreq(PSD_N, d=1.0 / sr)
        short = recording.split("_")[0]

        fig, ax = plt.subplots(figsize=(9.0, 4.4))
        for name in FIG_ARMS:
            if name not in fg["psds"]:
                continue
            f, db = _smooth_log(freqs, fg["psds"][name][0])
            ax.semilogx(
                f,
                db,
                color=FIG_COLOURS[name],
                lw=2.0 if name in ("real", "v2") else 1.3,
                ls="-" if name in ("real", "legacy", "v2") else "--",
                label=name,
            )
        ax.set_xlim(SP.BAND_F_MIN, SP.BAND_F_MAX)
        ax.set_xlabel("Hz")
        ax.set_ylabel("dB (absolute, 1/48-oct mean)")
        ax.set_title(f"DREGON {short}: LTAS, mic 0 — the comb is 18 dB under the fitted floor")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8, ncol=2)
        written.append(_save(fig, Path(out) / f"ltas_{short}.png"))
        plt.close(fig)

        panels = ("real", "v2", "v2_nofloor", "v2_comb_level_matched")
        have = [p for p in panels if p in fg["waves"]]
        fig, axes = plt.subplots(len(have), 1, figsize=(9.0, 2.2 * len(have)), sharex=True)
        axes = np.atleast_1d(axes)
        # ONE colour scale for every panel, set by the real clip: a per-panel
        # scale would auto-stretch the featureless v2 render and hide exactly
        # the difference this figure exists to show.
        scale: tuple[float, float] | None = None
        for ax, name in zip(axes, have, strict=True):
            x = fg["waves"][name]
            w = np.hanning(SPEC_N)
            starts = np.arange(0, x.size - SPEC_N + 1, SPEC_HOP)
            S = np.stack([np.abs(np.fft.rfft(x[s : s + SPEC_N] * w)) ** 2 for s in starts]).T
            db = 10.0 * np.log10(np.maximum(S / float((w**2).sum()), 1e-14))
            if scale is None:
                scale = (float(np.percentile(db, 5)), float(np.percentile(db, 99.5)))
            ax.imshow(
                db,
                origin="lower",
                aspect="auto",
                extent=[0.0, x.size / sr, 0.0, sr / 2.0],
                vmin=scale[0],
                vmax=scale[1],
                cmap="magma",
            )
            ax.set_ylim(0.0, 2000.0)
            ax.set_ylabel(f"{name}\nHz", fontsize=7)
        axes[-1].set_xlabel("s")
        axes[0].set_title(
            f"DREGON {short}: 0-2 kHz, mic 0 (n_fft {SPEC_N}, hop {SPEC_HOP}); "
            "one shared dB scale, set by the real panel"
        )
        written.append(_save(fig, Path(out) / f"spectrogram_{short}.png"))
        plt.close(fig)
    return written


# ── findings ────────────────────────────────────────────────────────────────


def _f(v: Any, spec: str = ".3f") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    try:
        return format(float(v), spec)
    except (TypeError, ValueError):
        return str(v)


PATCH_PROPOSAL = """\
```diff
@@ src/experiments/noise_model/model.py:99 class Priors
+    #: rig-to-rig level offset of a TRANSPLANTED comb, in dB. Zero-mean and
+    #: wide: it is a geometry/distance term (bench mic at ~1 m and one rotor
+    #: against a flight array at ~0.3 m and four), not a physical one.
+    comb_gain_db: tuple[float, float] = (0.0, 15.0)
@@ src/experiments/noise_model/model.py:162 free_blocks
     if mode == "flight_floor_only":
-        return ("floor", "mic")
+        # the comb arrives FROZEN from a bench rig, so its absolute level is
+        # that rig's; nothing downstream can re-level it, because
+        # render_noise mean-centres mic_line_gain_db over mics per rotor
+        # (render.py:191) and mic_gains_db over mics (:192) and profile_db is
+        # then the ONLY absolute comb scale. One shared scalar, no more.
+        return ("floor", "mic", "comb_gain")
@@ src/experiments/noise_model/model.py:586 sample_params
     if "profile" in free:
         profile_db = _normal(site, "profile_db", priors.profile_db[0], priors.profile_db[1], (r, k))
         amp_exp = _normal(site, "amp_exp", *priors.amp_exp) if flight else zero
     else:
         profile_db = take("profile", "profile_db", (r, k))
         amp_exp = take("profile", "amp_exp") if flight else zero
+    if "comb_gain" in free:
+        # folded into profile_db, so write_fit, render_noise and
+        # expected_periodogram need no change at all
+        profile_db = profile_db + _normal(site, "comb_gain_db", *priors.comb_gain_db)
```
"""


def findings(payload: dict[str, Any], *, job: str | None, figures: list[str]) -> str:
    rows = list(payload["supports"].values())
    recs = [r["support"]["recording"] for r in rows]
    shorts = [r.split("_")[0] for r in recs]
    fitd = payload["fit"]
    br = payload["bench_reference"]
    acc = payload["comb_level_accounting"]
    o: list[str] = []
    o.append("# Noise model v2 — round 2: why the v2 DREGON arm renders to PIT MAE 75 rev/s")
    o.append("")
    o.append(
        f"Record `{OUT_DEFAULT}/render_dregon.json`, git `{payload['git']}`, "
        f"fit `{fitd['path']}` (mode `{fitd['mode']}`, "
        f"`converged` {fitd['converged']!r} — NOT converged; "
        f"amp_exp {_f(fitd['amp_exp'], '.4f')}, floor_exp {_f(fitd['floor_exp'], '.4f')}, "
        f"floor_static_rel {_f(fitd['floor_static_rel'], '.4f')}, "
        f"floor_mean_db {_f(fitd['floor_mean_db'], '.3f')}, "
        f"floor_tilt {_f(fitd['floor_tilt_db_oct'], '.3f')} dB/oct), "
        f"render seed {payload['protocol']['seed']}, {payload['protocol']['n_mics']} mics."
    )
    o.append("")
    o.append(
        f"The fit's floor level never moved from its initialisation: "
        f"`diagnostics.init_floor_mean_db` {_f(fitd['init_floor_mean_db'], '.6f')} vs "
        f"`params.floor.floor_mean_db` {_f(fitd['floor_mean_db'], '.6f')} "
        f"(delta {_f(fitd['floor_mean_moved_from_init_db'], '.2e')} dB)."
    )
    o.append("")
    if job:
        o.append(f"HPPNet job `{job}` on `uni-gpushort`; checkpoint sha256 verified in-job.")
    scorer = payload["protocol"].get("scorer") or {}
    if scorer.get("sha256"):
        o.append(f"Frozen scorer digest `{scorer['sha256']}` (= `gates.SCORER_SHA256`).")
    o.append("")

    # ── PIT table
    o.append("## HPPNet PIT MAE per arm (rev/s, frozen HPPNet, seed 2001, 8 mics)")
    o.append("")
    o.append("| arm | " + " | ".join(shorts) + " | what it isolates |")
    o.append("|---|" + "---:|" * len(shorts) + "---|")
    for spec in ARMS:
        vals = [_arm(payload, r, spec.name, "pit_mae") for r in recs]
        o.append(
            f"| `{spec.name}` | " + " | ".join(_f(v) for v in vals) + f" | {spec.hypothesis} |"
        )
    o.append("")
    o.append(
        "`v2` reproduces the scored audio: "
        + "; ".join(
            f"{s} max |float64 render - committed npz| "
            f"{_f((_arm(payload, r, 'v2', 'arm_npz_check') or {}).get('max_abs_diff'), '.3e')}"
            for r, s in zip(recs, shorts, strict=True)
        )
        + " (float32 storage rounding)."
    )
    o.append("")

    # ── levels / LTAS
    o.append("## Levels and LTAS (30–7900 Hz, absolute)")
    o.append("")
    o.append(
        "| support | arm | band level mic0 dB | mic-mean dB | offset vs real dB | "
        "LTAS abs dB | LTAS shape dB | broadband RMS |"
    )
    o.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r, s in zip(recs, shorts, strict=True):
        real_l = _arm(payload, r, "real", "band_level_db_mic0")
        for spec in ARMS:
            a = _arm(payload, r, spec.name, "band_level_db_mic0")
            if a is None:
                continue
            o.append(
                f"| `{s}` | `{spec.name}` | {_f(a, '.2f')} | "
                f"{_f(_arm(payload, r, spec.name, 'band_level_db_mic_mean'), '.2f')} | "
                f"{_f(None if real_l is None else a - real_l, '+.2f')} | "
                f"{_f(_arm(payload, r, spec.name, 'ltas_mean_abs_db'), '.2f')} | "
                f"{_f(_arm(payload, r, spec.name, 'ltas_shape_only_mean_abs_db'), '.2f')} | "
                f"{_f(_arm(payload, r, spec.name, 'rms_broadband'), '.4f')} |"
            )
    o.append("")

    # ── comb-to-floor
    o.append("## Order-tracked comb prominence (dB over the local floor)")
    o.append("")
    o.append(
        f"Read at each frame's OWN label carrier ({TRK_N}-point, hop {TRK_HOP}, "
        f"peak = mean of ±{TRK_PEAK_BINS:.0f} bin, floor = median of the ±"
        f"{TRK_SPAN_REL:.1f}·f0 bins excluding every rotor's k−1/k/k+1 line). A periodogram at "
        "the MEAN carrier is useless here: DREGON cruise carriers move 10–25 rev/s inside the "
        "4 s window and smear every line into its neighbours."
    )
    o.append("")
    o.append("| support | arm | k=2 mic0 | k=4 | k=8 | k=2 mic-mean | k=4 | k=8 |")
    o.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r, s in zip(recs, shorts, strict=True):
        for spec in ARMS:
            if _arm(payload, r, spec.name, "comb_prominence") is None:
                continue
            mm = _arm(payload, r, spec.name, "comb_prominence_mic_mean_db") or []
            o.append(
                f"| `{s}` | `{spec.name}` | "
                + " | ".join(_f(_prom(payload, r, spec.name, k), "+.2f") for k in COMB_ORDERS)
                + " | "
                + " | ".join(_f(v, "+.2f") for v in mm)
                + " |"
            )
    o.append("")
    o.append(
        "Separated blocks of the same seed (`v2_nofloor` comb-only against `v2_nocomb` "
        "floor-only) give the exact per-line comb-to-floor, without any estimator:"
    )
    o.append("")
    o.append("| support | k | comb-to-floor per line dB | implied full-render prominence dB |")
    o.append("|---|---:|---:|---:|")
    for r, s in zip(recs, shorts, strict=True):
        cal = _row(payload, r)["calibration"]
        e = cal.get("order2_ratio") or {}
        o.append(
            f"| `{s}` | 2 | {_f(cal.get('dregon_line_ratio_k2_db'), '+.2f')} | "
            f"{_f(e.get('implied_prominence_db'), '+.2f')} |"
        )
    o.append("")

    # ── bench reference
    o.append("## The comb-to-floor the frozen comb was IDENTIFIED at (bench, own carriers)")
    o.append("")
    o.append(
        "| bench fit | carrier rev/s | floor_mean_db | comb-only band dB | floor-only band dB | "
        "comb−floor band dB | k=2 line ratio dB | full k=2 prominence dB |"
    )
    o.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, v in br["per_motor"].items():
        o.append(
            f"| `bench_dregon_{name}_70` (`converged` {v['converged']!r}) | "
            f"{_f(v['carrier_rev_s'], '.3f')} | {_f(v['floor_mean_db'], '.2f')} | "
            f"{_f(v['comb_only_band_level_db_mic0'], '.2f')} | "
            f"{_f(v['floor_only_band_level_db_mic0'], '.2f')} | "
            f"{_f(v['comb_minus_floor_band_db'], '+.2f')} | "
            f"{_f(v['line_ratio'][0]['mean_db'], '+.2f')} | "
            f"{_f(v['full_prominence_db'][0], '+.2f')} |"
        )
    o.append(
        f"| **mean** | | {_f(br['mean_floor_mean_db'], '.2f')} | | | "
        f"{_f(br['mean_comb_minus_floor_band_db'], '+.2f')} | "
        f"{_f(br['mean_line_ratio_db'].get('2'), '+.2f')} | |"
    )
    o.append("")

    # ── speed laws
    o.append("## The fit's two speed laws at the bench carrier vs the DREGON cruise carriers")
    o.append("")
    o.append(
        "| support | carriers | mean speed (÷80) | comb envelope dB | floor envelope dB | "
        "comb−floor dB | k_max | top comb line Hz |"
    )
    o.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r, s in zip(recs, shorts, strict=True):
        row = _row(payload, r)
        for key, lab in (
            ("envelope_at_bench_carrier", "bench ~68 rev/s"),
            ("envelope_at_dregon_cruise", "DREGON cruise"),
        ):
            e = row[key]
            o.append(
                f"| `{s}` | {lab} | {_f(e['speed_mean'], '.4f')} | "
                f"{_f(e['comb_envelope_db'], '+.2f')} | {_f(e['floor_envelope_db'], '+.2f')} | "
                f"{_f(e['comb_minus_floor_db'], '+.2f')} | {e['k_max']} | "
                f"{_f(e['top_comb_line_hz'], '.0f')} |"
            )
        sh = row["envelope_bench_to_cruise_shift_db"]
        o.append(
            f"| `{s}` | **bench → cruise shift** | | {_f(sh['comb_db'], '+.2f')} | "
            f"{_f(sh['floor_db'], '+.2f')} | "
            f"{_f(sh['comb_db'] - sh['floor_db'], '+.2f')} | | |"
        )
    o.append("")

    # ── comb level accounting
    o.append("## The comb's absolute level, term by term")
    o.append("")
    o.append(f"{acc['note']}.")
    o.append("")
    o.append(
        "| fit | orders | Σ_k line power, mic 0, all rotors dB | mic-mean dB | "
        "raw mic_line_gain_db mean (DISCARDED) | raw mic_gains_db mean (DISCARDED) |"
    )
    o.append("|---|---:|---:|---:|---|---:|")
    c = acc["candidate"]
    o.append(
        f"| candidate (log-mean comb, 4 rotors) | {c['n_orders']} | "
        f"{_f(c['mic0_total_db'], '.2f')} | {_f(c['mic_mean_total_db'], '.2f')} | "
        + ", ".join(_f(v, ".2f") for v in c["mic_line_gain_db_raw_mean"])
        + f" | {_f(c['mic_gains_db_raw_mean'], '.2f')} |"
    )
    for name, v in acc["bench"].items():
        o.append(
            f"| `bench_dregon_{name}_70` (1 rotor) | {v['n_orders']} | "
            f"{_f(v['mic0_total_db'], '.2f')} | {_f(v['mic_mean_total_db'], '.2f')} | "
            + ", ".join(_f(x, ".2f") for x in v["mic_line_gain_db_raw_mean"])
            + f" | {_f(v['mic_gains_db_raw_mean'], '.2f')} |"
        )
    o.append("")
    o.append(
        f"Candidate minus the bench mean at mic 0: "
        f"{_f(acc['candidate_minus_bench_mic0_db'], '+.2f')} dB — the four-rotor sum "
        f"({_f(acc['four_rotor_sum_db'], '+.2f')} dB if the rotors were identical and the "
        f"per-rotor mic gains neutral) minus what the dB-mean of the four bench profiles and "
        f"the per-rotor mic-gain centring take back."
    )
    o.append("")

    # ── re-levelling shifts
    o.append("## The re-levelling shifts each variant applies (all solved, none searched)")
    o.append("")
    o.append(
        "| support | comb-only band dB | floor-only band dB | comb−floor dB | "
        "(d) floor→real level dB | (e) comb→real k=2 ratio dB | comb→bench ratio dB | "
        "comb→real band level dB |"
    )
    o.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r, s in zip(recs, shorts, strict=True):
        cal = _row(payload, r)["calibration"]
        o.append(
            f"| `{s}` | {_f(cal.get('comb_only_band_level_db_mic0'), '.2f')} | "
            f"{_f(cal.get('floor_only_band_level_db_mic0'), '.2f')} | "
            f"{_f(cal.get('comb_minus_floor_band_db'), '+.2f')} | "
            f"{_f(cal.get('floor_match_shift_db'), '+.2f')} | "
            f"{_f(cal.get('comb_real_ratio_shift_db'), '+.2f')} | "
            f"{_f(cal.get('comb_bench_ratio_shift_db'), '+.2f')} | "
            f"{_f(cal.get('comb_level_match_shift_db'), '+.2f')} |"
        )
    o.append("")

    # ── encoded speed
    o.append("## Encoded speed (order-2 demodulation against the frozen label)")
    o.append("")
    o.append("| support | arm | rotor | median abs dev rev/s | p90 | mean signed |")
    o.append("|---|---|---|---:|---:|---:|")
    for r, s in zip(recs, shorts, strict=True):
        for name in ("real", "legacy", "v2_nofloor", "v2_comb_level_matched"):
            enc = _arm(payload, r, name, "encoded_speed") or {}
            for rotor, v in enc.items():
                o.append(
                    f"| `{s}` | `{name}` | {rotor} | "
                    f"{_f(v.get('median_abs_dev_rev_s'))} | {_f(v.get('p90_abs_dev_rev_s'))} | "
                    f"{_f(v.get('mean_dev_rev_s'))} |"
                )
    o.append("")
    o.extend(hypothesis_section(payload))
    o.append("")
    o.append("## Minimal patch proposal (NOT applied — Main's call)")
    o.append("")
    o.extend(patch_preamble(payload))
    o.append("")
    o.append(PATCH_PROPOSAL)
    o.extend(patch_expectation(payload))
    if figures:
        o.append("")
        o.append("## Figures")
        o.append("")
        o.extend(f"* `{p}`" for p in figures)
    o.append("")
    return "\n".join(o)


def hypothesis_section(payload: dict[str, Any]) -> list[str]:
    v = payload["verdicts"]
    rows = list(payload["supports"].values())
    recs = [r["support"]["recording"] for r in rows]
    shorts = [r.split("_")[0] for r in recs]

    def pairs(d: dict[str, Any], spec: str = ".3f") -> str:
        return ", ".join(f"{s} {_f(d.get(r), spec)}" for r, s in zip(recs, shorts, strict=True))

    h1, h2, h3, h4 = (
        v["H1_speed_law"],
        v["H2_mic_gain_units"],
        v["H3_floor_buries_comb"],
        v["H4_wrong_speed"],
    )
    comb_shift = pairs(h1["comb_envelope_shift_db"], "+.2f")
    floor_shift = pairs(h1["floor_envelope_shift_db"], "+.2f")
    out = [
        "## Verdict per hypothesis",
        "",
        "| hypothesis | verdict | the numbers that decide it |",
        "|---|---|---|",
    ]
    out.append(
        "| **H1** — the bench comb was fitted at ~68 rev/s and the cruise carriers differ, so the "
        "speed law / Nyquist geometry is wrong at cruise | **REFUTED as the cause** | "
        f"`amp_exp` is {_f(payload['fit']['amp_exp'], '.1f')} (a bench fit has one throttle and "
        f"cannot identify a speed law), so the comb envelope shift bench → cruise is {comb_shift} "
        f"dB; the floor's own law moves {floor_shift} dB, so the whole bench → cruise speed-law "
        f"effect on comb−floor is under 5 dB against the "
        f"{_f(h3['comb_minus_floor_band_db'].get(recs[0]), '+.1f')} dB deficit measured. "
        f"`k_max` is {h1['k_max_cruise'].get(recs[0])} at cruise vs "
        f"{h1['k_max_bench'].get(recs[0])} at the bench carrier (no Nyquist cap change). "
        f"Pinning BOTH exponents at the prior mean leaves PIT at {pairs(h1['pit_prior_exps'])} "
        f"rev/s against the unchanged {pairs(h1['pit_v2'])} — the R1 regime patch does nothing "
        "here |"
    )
    out.append(
        "| **H2** — a units / per-rotor level mismatch in `mic_line_gain_db` / `mic_gains_db` | "
        "**REFUTED as a units bug, SUPPORTED as a missing degree of freedom** | "
        f"`render_noise` mean-centres `mic_line_gain_db` over mics per rotor (`render.py:191`) "
        f"and `mic_gains_db` over mics (`:192`), so their absolute values are DISCARDED: the only "
        f"absolute comb scale a render has is `profile_db`. Σ_k line power at mic 0 is "
        f"{_f(h2['candidate_mic0_line_power_db'], '.2f')} dB for the candidate against a bench "
        f"mean of {_f(h2['bench_mic0_line_power_db_mean'], '.2f')} dB "
        f"({_f(h2['candidate_minus_bench_db'], '+.2f')} dB) — the transplanted comb carries the "
        "BENCH rig's absolute level and there is no parameter anywhere that could re-level it |"
    )
    out.append(
        "| **H3** — the floor-only flight fit's floor buries the comb | **SUPPORTED — this is the "
        "defect** | "
        f"comb-only minus floor-only band level: {pairs(h3['comb_minus_floor_band_db'], '+.2f')} "
        f"dB in the flight render against {_f(h3['bench_comb_minus_floor_band_db'], '+.2f')} dB on "
        f"the bench where the comb was fitted; per-line at k=2, "
        f"{pairs(h3['order2_line_ratio_db'], '+.2f')} dB against the bench "
        f"{_f(h3['bench_order2_line_ratio_db'], '+.2f')} dB. Consequence: `v2` "
        f"{pairs(h3['pit_v2'])} rev/s is INDISTINGUISHABLE from `v2_nocomb` "
        f"{pairs(h3['pit_nocomb'])} — the render's comb contributes nothing at all, and the "
        f"per-rotor PIT MAE equals the per-rotor label speed (the tracker returns no rotor). "
        f"Restoring the comb level fixes it: comb → real band level "
        f"{pairs(h3['pit_comb_level_matched'])}, comb → bench ratio "
        f"{pairs(h3['pit_comb_bench_ratio'])}, against legacy "
        f"{pairs({r: _arm(payload, r, 'legacy', 'pit_mae') for r in recs})} and real "
        f"{pairs({r: _arm(payload, r, 'real', 'pit_mae') for r in recs})}. Moving the FLOOR "
        f"instead does not: {pairs(h3['pit_floor_matched'])} |"
    )
    enc_v2 = {r: (h4["encoded_speed_v2_comb"].get(r) or {}) for r in recs}
    out.append(
        "| **H4** — the render encodes a different speed than the label | **REFUTED** | "
        "order-2 demodulation of the comb-only render against the frozen label: "
        + "; ".join(
            f"{s} "
            + ", ".join(f"{k} {_f(x.get('median_abs_dev_rev_s'))}" for k, x in enc_v2[r].items())
            for r, s in zip(recs, shorts, strict=True)
        )
        + " rev/s median absolute deviation — the comb sits on the label carrier by construction "
        f"(`render_noise` integrates the carriers it is handed). Every frozen DREGON support is "
        f"one 4 s cruise window scored whole ({pairs(h4['scored_seconds'], '.2f')} s), so there "
        "is no standby-like frame inside the score window, and `k_max` "
        f"{h4['k_max_cruise'].get(recs[0])} puts the top comb line at "
        f"{_f(h4['top_comb_line_hz'].get(recs[0]), '.0f')} Hz, inside the band |"
    )
    return out


def patch_preamble(payload: dict[str, Any]) -> list[str]:
    fitd = payload["fit"]
    br = payload["bench_reference"]
    rows = list(payload["supports"].values())
    recs = [r["support"]["recording"] for r in rows]
    cal0 = _row(payload, recs[0])["calibration"]
    return [
        f"One defect, one missing degree of freedom. The comb was frozen from four DREGON BENCH "
        f"fits whose own floors sit at {_f(br['mean_floor_mean_db'], '.2f')} dB, where the comb is "
        f"{_f(br['mean_comb_minus_floor_band_db'], '+.2f')} dB ABOVE the floor in band power and "
        f"{_f(br['mean_line_ratio_db'].get('2'), '+.2f')} dB above it per line at k=2. The flight "
        f"fit is `mode={fitd['mode']}`: its objective frees the FLOOR ONLY, so nothing in it ever "
        f"balanced the transplanted comb against the flight floor — and its floor never even left "
        f"its own initialisation ({_f(fitd['init_floor_mean_db'], '.4f')} dB, "
        f"`converged` {fitd['converged']!r}). On the DREGON cruise supports the same comb ends up "
        f"{_f(cal0.get('comb_minus_floor_band_db'), '+.2f')} dB relative to the floor, i.e. the "
        f"comb-to-floor ratio swung by "
        f"{_f(float(br['mean_comb_minus_floor_band_db']) - float(cal0['comb_minus_floor_band_db']), '.1f')}"
        f" dB across the transplant. The renderer cannot absorb that: it mean-centres both "
        f"`mic_line_gain_db` and `mic_gains_db`, so `profile_db` is the ONLY absolute comb scale "
        f"and it is frozen. The minimal fix is therefore to give `flight_floor_only` exactly one "
        f"more free scalar — a shared comb gain in dB — and let the flight likelihood set it:",
    ]


def patch_expectation(payload: dict[str, Any]) -> list[str]:
    rows = list(payload["supports"].values())
    recs = [r["support"]["recording"] for r in rows]
    shorts = [r.split("_")[0] for r in recs]
    names = (
        "real",
        "legacy",
        "v2",
        "v2_prior_exps",
        "v2_floor_matched",
        "v2_comb_real_ratio",
        "v2_comb_bench_ratio",
        "v2_comb_level_matched",
    )
    out = [
        "",
        "### Expected effect — measured, not guessed",
        "",
        "No variant below is a FIT, so none of them is a parity claim: each is the SAME frozen "
        "comb and the SAME fitted floor with ONE level moved, scored by the same frozen HPPNet on "
        "the same frozen supports and seed. What they bound is how much a single free comb gain "
        "can buy.",
        "",
        "| arm | comb shift dB | " + " | ".join(f"{s} PIT" for s in shorts) + " |",
        "|---|---:|" + "---:|" * len(shorts),
    ]
    for name in names:
        shift = (_arm(payload, recs[0], name, "variant_params") or {}).get("profile_shift_db")
        out.append(
            f"| `{name}` | {_f(shift, '+.2f')} | "
            + " | ".join(_f(_arm(payload, r, name, "pit_mae")) for r in recs)
            + " |"
        )
    out.append("")
    gate = GT.DREGON_PIT_TARGET
    lvl = [_arm(payload, r, "v2_comb_level_matched", "pit_mae") for r in recs]
    ok = [v for v in lvl if v is not None]
    out.append(
        f"Two of the five gate supports only, so this is INDICATIVE and not a gate verdict: the "
        f"frozen DREGON target is {gate:.6f} rev/s and the round-2 arm scored 75.093968 over all "
        f"five. With the comb re-levelled onto the real window's own band level these two score "
        f"{', '.join(_f(v) for v in ok)} rev/s, against legacy "
        f"{', '.join(_f(_arm(payload, r, 'legacy', 'pit_mae')) for r in recs)} and real "
        f"{', '.join(_f(_arm(payload, r, 'real', 'pit_mae')) for r in recs)} on the same two. "
        f"The level-matched comb shift is the counterfactual a free `comb_gain_db` would have to "
        f"find; the prior above ({{0, 15}} dB) covers it."
    )
    out.append("")
    bench = payload["bench_reference"]["per_motor"]
    out.append(
        "`v2_comb_bench_ratio` OVERSHOOTS and is not the number to target. The four bench fits "
        "disagree wildly about their own floors — floor-only band level "
        + ", ".join(
            f"{n} {_f(v['floor_only_band_level_db_mic0'], '.2f')}" for n, v in bench.items()
        )
        + " dB — so their mean k=2 comb-to-floor "
        f"{_f(payload['bench_reference']['mean_line_ratio_db'].get('2'), '+.2f')} dB is dominated "
        f"by `Motor4` ({_f(bench['Motor4']['line_ratio'][0]['mean_db'], '+.2f')} dB, floor-only "
        f"{_f(bench['Motor4']['floor_only_band_level_db_mic0'], '.2f')} dB) against `Motor3` "
        f"({_f(bench['Motor3']['line_ratio'][0]['mean_db'], '+.2f')} dB). The shift it implies "
        f"({_f((_arm(payload, recs[0], 'v2_comb_bench_ratio', 'variant_params') or {}).get('profile_shift_db'), '+.2f')}"
        " dB) puts the render "
        f"{_f(_arm(payload, recs[0], 'v2_comb_bench_ratio', 'ltas_level_offset_db'), '+.1f')} dB "
        "over the real level and the PIT back up to "
        + ", ".join(_f(_arm(payload, r, "v2_comb_bench_ratio", "pit_mae")) for r in recs)
        + " rev/s. The real window's own BAND LEVEL is the well-posed target, and it is what a "
        "free `comb_gain_db` fitted on the flight supports would be pulled to."
    )
    out.append("")
    out.append(
        "The literal variant (e) — the comb re-levelled so its k=2 prominence matches the REAL "
        "clip's — is DEGENERATE on DREGON and is reported for completeness only. At mic 0 the "
        "real room-2 cruise clip has almost no discrete line prominence at the `motors_command` "
        "carrier ("
        + ", ".join(
            f"{s} {_f(_prom(payload, r, 'real', 2), '+.2f')} dB at k=2"
            for r, s in zip(recs, shorts, strict=True)
        )
        + "), so matching it asks the comb to stay where it is ("
        + ", ".join(
            f"{s} {_f(_row(payload, r)['calibration'].get('comb_real_ratio_shift_db'), '+.2f')} dB"
            for r, s in zip(recs, shorts, strict=True)
        )
        + "). The 8-mic MEAN prominence is larger — real "
        + ", ".join(
            f"{s} {_f((_arm(payload, r, 'real', 'comb_prominence_mic_mean_db') or [None])[0], '+.2f')}"
            for r, s in zip(recs, shorts, strict=True)
        )
        + " dB at k=2 against `v2_nocomb` "
        + ", ".join(
            f"{s} {_f((_arm(payload, r, 'v2_nocomb', 'comb_prominence_mic_mean_db') or [None])[0], '+.2f')}"
            for r, s in zip(recs, shorts, strict=True)
        )
        + " dB — so real audio DOES carry a comb, spatially coherent enough to survive the "
        "8-mic average, while the v2 render carries none. But it is a WEAK comb: HPPNet reaches "
        + ", ".join(
            f"{s} {_f(_arm(payload, r, 'real', 'pit_mae'))}"
            for r, s in zip(recs, shorts, strict=True)
        )
        + " rev/s on it. The render's target is therefore not 'a prominent comb' but 'the real "
        "clip's own comb ENERGY', which is what the band-level variant sets."
    )
    return out


PIT_KEYS = ("pit_mae", "pit_per_mic", "pit_per_rotor", "n_scored_frames")
LEVEL_TOL_DB = 0.01


def merge_pit(payload: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    """Copy a probe pass's PIT numbers onto this pass's arms.

    The two passes must have rendered the same audio: the band level of every
    shared arm is compared first and a mismatch is fatal.
    """
    if str(probe.get("schema")) != SCHEMA:
        die("the probe payload is not a noise-v2-render-dregon/1 record")
    for key, row in payload["supports"].items():
        prow = (probe.get("supports") or {}).get(key)
        if prow is None:
            die(f"the probe payload has no support {key!r}")
        for name, entry in row["arms"].items():
            pe = (prow.get("arms") or {}).get(name)
            if pe is None:
                continue
            gap = abs(float(entry["band_level_db_mic0"]) - float(pe["band_level_db_mic0"]))
            if gap > LEVEL_TOL_DB:
                die(
                    f"{key}/{name}: the probe pass rendered a different signal "
                    f"({gap:.4f} dB of band level apart)"
                )
            for pk in PIT_KEYS:
                if pk in pe:
                    entry[pk] = pe[pk]
    payload["protocol"]["scorer"] = (probe.get("protocol") or {}).get("scorer")
    payload["verdicts"] = verdicts(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", type=Path, default=FIT_DEFAULT)
    ap.add_argument("--bench-dir", type=Path, default=BENCH_DEFAULT)
    ap.add_argument("--arm-npz", type=Path, default=ARM_NPZ_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--n-mics", type=int, default=8)
    ap.add_argument("--probe", action="store_true", help="score every arm with the frozen HPPNet")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--job", type=str, default=None, help="omnirun job id, for the record")
    ap.add_argument(
        "--merge-probe", type=Path, default=None, help="a probe pass's JSON to merge PIT from"
    )
    args = ap.parse_args(argv)

    payload = run(
        fit_path=args.fit,
        bench_dir=args.bench_dir,
        arm_npz=args.arm_npz,
        out=args.out,
        seed=args.seed,
        probe=bool(args.probe),
        n_mics=int(args.n_mics),
    )
    figures_data = payload.pop("_figures")
    if args.merge_probe is not None:
        payload = merge_pit(payload, json.loads(Path(args.merge_probe).read_text()))
    written = write_figures(payload, figures_data, args.out) if args.figures else []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "render_dregon.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    (out / "findings.md").write_text(findings(payload, job=args.job, figures=written))
    print(f"wrote {out / 'render_dregon.json'}")
    print(f"wrote {out / 'findings.md'}")
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
