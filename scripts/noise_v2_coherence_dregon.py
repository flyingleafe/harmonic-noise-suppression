"""Is it the comb's STATISTICS, not its level, that HPPNet locks onto?

`round3/dregon_humps/render_vs_model.md` closed the level question: the v2
render reproduces its own forward model cell for cell (I/M within 0.05 dB in
every cell class) and the fit reproduces the real window's per-cell power to a
uniform +2 to +4 dB across comb and floor cells alike — and HPPNet still scores
73 rev/s on the render against 0.89 on the real clip. Whatever the tracker
keys on is therefore not per-cell comb POWER.

This runner holds the comb's power spectrum EXACTLY at the fitted ``M`` and
varies only the phase structure of the lines:

* ``C1`` legacy-style coherence split: the share ``w_k = exp(-(k/k_half)^2)``
  of each order's power stays a tone and the rest becomes narrowband noise of
  the legacy export's own pedestal half-width — Rayleigh cells instead of a
  steady tone, at the same mean power;
* ``C2`` cross-ORDER phase lock: every order of a rotor shares one phase at
  each microphone (a periodic blade-passage waveform) with the per-line ``psi``
  diffusion and the shaft state as fitted;
* ``C3`` cross-MIC lock: one phase per (rotor, order), identical on all eight
  microphones;
* ``C4`` both; ``C5`` = C4 with ``gamma_rk = 0`` for ``k <= 8``, the low orders
  fully shaft-locked.

The renderer here is :func:`render_phase`, a copy of
``noise_model.render.render_noise`` with the per-(mic, rotor, order) phase draw
made configurable. With every option off it reproduces that function to float
round-off — the payload records the difference — so an arm's PIT can only
differ through the phase structure the arm names.

Two statistics say where the real window sits: the inter-order phase coherence
of the demodulated low orders and the inter-microphone magnitude-squared
coherence, both on the label's own carriers.

    python scripts/noise_v2_coherence_dregon.py --probe --out DIR
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from experiments.noise_model import gates as GT
from experiments.noise_model import model as MD
from experiments.noise_model import render as RD
from experiments.noise_model import spectrum as SP
from experiments.stochastic_fit import revised_eval as RE

SCHEMA = "noise-v2-dregon-coherence/1"
OUT_DEFAULT = Path("results/noise_v2/rounds/round3/dregon_humps")
FIT_DEFAULT = Path("results/noise_v2/rounds/round3/fits/dregon_room2_floor__flight_floor_lowk.json")
RECORDINGS = (
    "free-flight_nosource_room2",
    "hovering_nosource_room2",
    "updown_nosource_room2",
)

#: The legacy DREGON cruise export's own split, measured in `round3/dregon_humps`.
LEGACY_K_HALF = 1.6962707042694292
LEGACY_GAMMA0_HZ = 13.068
LEGACY_GAMMA_SLOPE_HZ = 0.211

#: Demodulation front end for the coherence statistics: +-16 Hz around each
#: k f_r(t), in the time domain, so no STFT phase convention enters.
DEMOD_BW_HZ = 16.0
DEMOD_HOP = 128
COH_ORDERS = (1, 2, 3, 4, 5, 6)
MSC_ORDERS = (1, 2)


def die(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def _module(name: str) -> Any:
    path = Path("scripts") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        die(f"cannot load {path}")
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


# ── the renderer, with the phase structure exposed ──────────────────────────


def render_phase(
    fit: dict[str, Any],
    rps_rev_s: np.ndarray,
    *,
    sr: int = SP.FLIGHT_SR,
    n_mics: int = 8,
    seed: int = 0,
    sr_work: int = SP.SAMPLE_RATE_WORK,
    order_lock: bool = False,
    mic_lock: bool = False,
    zero_gamma_k: int = 0,
    coherence_k_half: float = 0.0,
) -> np.ndarray:
    """``noise_model.render.render_noise`` with the line phases configurable.

    ``order_lock`` gives every order of a rotor the same microphone phase (a
    periodic blade-passage waveform); ``mic_lock`` gives every microphone the
    same phase; ``zero_gamma_k`` pins ``gamma_rk = 0`` up to that order;
    ``coherence_k_half`` > 0 splits each order's power into a tone of share
    ``w_k = exp(-(k/k_half)^2)`` and a narrowband-NOISE remainder of Lorentzian
    half-width ``LEGACY_GAMMA0_HZ + LEGACY_GAMMA_SLOPE_HZ k`` — the legacy
    coherence split, at the same total power.

    Every draw is taken in the same order and the same shape as the original,
    so with all options off the output is that function's, bit for bit.
    """
    from scipy.signal import lfilter

    from experiments.stochastic_fit import clips as C
    from experiments.stochastic_fit.data import Clip

    p = RD._check_schema(fit)
    rps = np.atleast_2d(np.asarray(rps_rev_s, dtype=np.float64))
    n_rotors, n_out = rps.shape
    profile_db = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
    if profile_db.shape[0] == 1:
        profile_db = np.repeat(profile_db, n_rotors, axis=0)
    gamma_hz = MD.gamma_from_params(p)
    if gamma_hz.shape[0] == 1:
        gamma_hz = np.repeat(gamma_hz, n_rotors, axis=0)

    oversample = int(sr_work) // int(sr)
    n_work = n_out * oversample
    dt = 1.0 / float(sr_work)
    t_src = np.arange(n_out) / float(sr)
    t_work = np.arange(n_work) / float(sr_work)
    f0 = np.maximum(np.stack([np.interp(t_work, t_src, r) for r in rps]), RD.SPEED_FLOOR_RPS)
    k_max = min(
        int(profile_db.shape[1]),
        SP.k_max_for_carrier(f0.max(axis=1), sr, k_cap=int(profile_db.shape[1])),
    )
    ss = np.random.SeedSequence(int(seed))
    rng_state, rng_psi, rng_alpha, rng_floor = (np.random.default_rng(s) for s in ss.spawn(4))
    lam = float(p["lam"])
    sigma_nu = float(p["sigma_nu"])
    innov = rng_state.standard_normal((n_rotors, n_work, 2))
    theta, _nu = RD.simulate_state(innov, lam=lam, sigma=sigma_nu, dt=dt)
    phase = 2.0 * np.pi * np.cumsum(f0, axis=1) * dt + theta

    mic_line_db = np.asarray(p["profile"]["mic_line_gain_db"], dtype=np.float64)
    mic_floor_db = np.asarray(p["floor"]["mic_floor_db"], dtype=np.float64)
    gain_all_db = np.asarray(p["mic_gains_db"], dtype=np.float64)
    mic_line_db = mic_line_db[:n_mics, :n_rotors] if mic_line_db.ndim == 2 else mic_line_db[:n_mics]
    line_gain = 10.0 ** ((mic_line_db - mic_line_db.mean(axis=0, keepdims=True)) / 10.0)
    all_gain = 10.0 ** ((gain_all_db[:n_mics] - gain_all_db[:n_mics].mean()) / 10.0)

    amp_exp = float(p["profile"]["amp_exp"])
    speed = f0 / RD.AMP_RPS_REF
    audio = np.zeros((n_mics, n_work), dtype=np.float64)
    rng_inc = np.random.default_rng(int(seed) + 991)
    for r in range(n_rotors):
        line_amp = np.sqrt(2.0 * 10.0 ** (profile_db[r, :k_max] / 10.0))
        speed_amp = np.sqrt(speed[r] ** amp_exp)
        alpha_first: np.ndarray | None = None
        for k in range(1, k_max + 1):
            gamma = float(gamma_hz[r, k - 1])
            if zero_gamma_k and k <= int(zero_gamma_k):
                gamma = 0.0
            step = math.sqrt(4.0 * math.pi * max(gamma, 0.0) * dt)
            psi = np.cumsum(rng_psi.standard_normal(n_work) * step)
            arg = k * phase[r] + psi
            env = line_amp[k - 1] * speed_amp
            alpha = rng_alpha.uniform(0.0, 2.0 * np.pi, size=n_mics)
            if alpha_first is None:
                alpha_first = alpha.copy()
            if order_lock:
                alpha = alpha_first.copy()
            if mic_lock:
                alpha = np.full(n_mics, float(alpha[0]))
            w = 1.0
            if coherence_k_half > 0.0:
                w = float(
                    np.clip(math.exp(-((k / float(coherence_k_half)) ** 2)), 1e-6, 1.0 - 1e-6)
                )
            ec = (env * math.sqrt(w)) * np.cos(arg)
            es = (env * math.sqrt(w)) * np.sin(arg)
            for m in range(n_mics):
                g = math.sqrt(line_gain[m, r]) if line_gain.ndim == 2 else math.sqrt(line_gain[m])
                audio[m] += (g * math.cos(alpha[m])) * ec
                audio[m] -= (g * math.sin(alpha[m])) * es
            if coherence_k_half > 0.0 and w < 1.0 - 1e-9:
                # the incoherent share: the same line, its envelope a complex
                # OU process of Lorentzian half-width gamma_ped and unit mean
                # power, so the mean spectrum is unchanged and only the cell
                # statistics (steady tone -> Rayleigh) change.
                gped = LEGACY_GAMMA0_HZ + LEGACY_GAMMA_SLOPE_HZ * float(k)
                a = math.exp(-2.0 * math.pi * gped * dt)
                drive = rng_inc.standard_normal((2, n_work)) * math.sqrt((1.0 - a * a) / 2.0)
                gz = lfilter([1.0], [1.0, -a], drive, axis=-1)
                env_i = env * math.sqrt(1.0 - w)
                ic = env_i * (gz[0] * np.cos(arg) - gz[1] * np.sin(arg))
                is_ = env_i * (gz[0] * np.sin(arg) + gz[1] * np.cos(arg))
                for m in range(n_mics):
                    g = (
                        math.sqrt(line_gain[m, r])
                        if line_gain.ndim == 2
                        else math.sqrt(line_gain[m])
                    )
                    audio[m] += (g * math.cos(alpha[m])) * ic
                    audio[m] -= (g * math.sin(alpha[m])) * is_

    shape_mat, tilt_oct = RD.floor_geometry(np.fft.rfftfreq(n_work, d=dt), SP.floor_ctrl_hz(sr))
    shape_db = SP.floor_shape_db(p["floor"]["floor_shape_z"], sr=sr)
    floor_psd = RD.floor_power_spectrum(
        shape_mat,
        tilt_oct,
        mean_db=float(p["floor"]["floor_mean_db"]),
        ctrl_db=shape_db,
        tilt_db_oct=float(p["floor"]["floor_tilt_db_oct"]),
        rate_factor=float(sr_work) / float(sr),
    )
    floor_gain_t = (speed ** float(p["floor"]["floor_exp"])).mean(axis=0) + float(
        p["floor"]["floor_static_rel"]
    )
    floor_amp = np.sqrt(floor_psd)
    for m in range(n_mics):
        white = rng_floor.standard_normal(n_work)
        shaped = np.fft.irfft(np.fft.rfft(white) * floor_amp, n=n_work)
        audio[m] += shaped * np.sqrt(floor_gain_t) * math.sqrt(10.0 ** (mic_floor_db[m] / 10.0))
    audio *= np.sqrt(all_gain)[:, None]
    out = C.decimate(
        Clip("noise_v2_render", "synthetic", RD.antialias(audio, sr_work), f0, int(sr_work)),
        int(sr),
    )
    return np.asarray(out.audio, dtype=np.float64)[:, :n_out]


# ── the two statistics ──────────────────────────────────────────────────────


def demodulate(
    audio: np.ndarray, f0_track: np.ndarray, k: int, *, sr: int, bw_hz: float = DEMOD_BW_HZ
) -> np.ndarray:
    """``(mics, frames)`` complex envelope of order ``k`` on ONE rotor's label.

    Time-domain demodulation by the label's own cumulative phase, low-passed to
    ``+-bw_hz`` by zeroing the spectrum: no STFT phase convention enters, and a
    line that rides the label exactly comes out as a constant.
    """
    x = np.asarray(audio, dtype=np.float64)
    n = int(x.shape[-1])
    ph = 2.0 * np.pi * np.cumsum(np.asarray(f0_track, dtype=np.float64)) / float(sr)
    lo = np.exp(-1j * float(k) * ph)
    freqs = np.fft.fftfreq(n, d=1.0 / float(sr))
    keep = np.abs(freqs) <= float(bw_hz)
    out = np.fft.ifft(np.fft.fft(x * lo[None, :], axis=-1) * keep[None, :], axis=-1)
    return out[:, ::DEMOD_HOP]


def coherence_stats(audio: np.ndarray, f0_tracks: np.ndarray, *, sr: int) -> dict[str, Any]:
    """Inter-ORDER phase coherence and inter-MIC magnitude-squared coherence."""
    f0 = np.asarray(f0_tracks, dtype=np.float64)
    n_rotors = int(f0.shape[0])
    z: dict[tuple[int, int], np.ndarray] = {}
    for r in range(n_rotors):
        for k in COH_ORDERS:
            z[(r, k)] = demodulate(audio, f0[r], k, sr=sr)
    inter_order: dict[str, list[float]] = {}
    for k in COH_ORDERS[:-1]:
        vals: list[float] = []
        for r in range(n_rotors):
            a, b = z[(r, k)], z[(r, k + 1)]
            u = a * np.conj(b)
            mag = np.abs(u)
            ok = mag > 0
            for m in range(int(a.shape[0])):
                sel = ok[m]
                if int(sel.sum()) < 8:
                    continue
                vals.append(float(np.abs(np.mean(u[m][sel] / mag[m][sel]))))
        inter_order[f"k{k}_{k + 1}"] = vals
    msc: dict[str, list[float]] = {}
    for k in MSC_ORDERS:
        vals = []
        for r in range(n_rotors):
            a = z[(r, k)]
            n_m = int(a.shape[0])
            for m in range(n_m):
                for mm in range(m + 1, n_m):
                    num = abs(complex(np.mean(a[m] * np.conj(a[mm])))) ** 2
                    den = float(np.mean(np.abs(a[m]) ** 2)) * float(np.mean(np.abs(a[mm]) ** 2))
                    if den > 0:
                        vals.append(float(num / den))
        msc[f"k{k}"] = vals
    return dict(
        inter_order_phase_coherence={k: float(np.mean(v)) for k, v in inter_order.items() if v},
        inter_order_mean=float(
            np.mean([x for v in inter_order.values() for x in v]) if inter_order else float("nan")
        ),
        inter_mic_msc={k: float(np.mean(v)) for k, v in msc.items() if v},
        protocol=dict(bw_hz=DEMOD_BW_HZ, hop=DEMOD_HOP, orders=list(COH_ORDERS)),
    )


# ── the arms ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Arm:
    name: str
    tag: str
    label: str
    kind: str = "v2"
    order_lock: bool = False
    mic_lock: bool = False
    zero_gamma_k: int = 0
    coherence_k_half: float = 0.0


ARMS: tuple[Arm, ...] = (
    Arm("real", "real", "the real DREGON room-2 clip", kind="real"),
    Arm("legacy", "legacy", "legacy stage-2 baseline, identity-matched", kind="legacy"),
    Arm("v2", "v2", "the round-3 v2 candidate, rendered by render_phase with every option off"),
    Arm(
        "c1_split",
        "C1",
        f"legacy-style coherence split (k_half {LEGACY_K_HALF:.3f}, pedestal "
        f"{LEGACY_GAMMA0_HZ:g}+{LEGACY_GAMMA_SLOPE_HZ:g}k Hz), same total power per order",
        coherence_k_half=LEGACY_K_HALF,
    ),
    Arm("c2_order", "C2", "cross-ORDER phase lock: one phase per (mic, rotor)", order_lock=True),
    Arm("c3_mic", "C3", "cross-MIC lock: one phase per (rotor, order)", mic_lock=True),
    Arm("c4_both", "C4", "cross-order AND cross-mic lock", order_lock=True, mic_lock=True),
    Arm(
        "c5_both_shaftlock",
        "C5",
        "C4 plus gamma_rk = 0 for k <= 8",
        order_lock=True,
        mic_lock=True,
        zero_gamma_k=8,
    ),
)


def run(
    *, fit_path: Path, out: Path, seed: int, probe: bool, n_mics: int, recordings: tuple[str, ...]
) -> dict[str, Any]:
    rs = _module("noise_v2_round_score")
    fit = json.loads(Path(fit_path).read_text())
    probe_obj = rs.Probe.load() if probe else None
    legacy = rs.legacy_arm("dregon")
    payload: dict[str, Any] = dict(
        schema=SCHEMA,
        git=git_rev(),
        fit=dict(path=str(fit_path), mode=fit.get("mode")),
        protocol=dict(
            seed=int(seed),
            n_mics=int(n_mics),
            demod=dict(bw_hz=DEMOD_BW_HZ, hop=DEMOD_HOP),
            legacy_split=dict(
                k_half=LEGACY_K_HALF,
                gamma0_hz=LEGACY_GAMMA0_HZ,
                gamma_slope_hz=LEGACY_GAMMA_SLOPE_HZ,
            ),
            scorer=(probe_obj.record if probe_obj is not None else None),
            arms=[dict(name=a.name, tag=a.tag, label=a.label) for a in ARMS],
        ),
        supports={},
    )
    for recording in recordings:
        found = [s for s in GT.DREGON_CRUISE_SUPPORTS if s.recording == recording]
        if not found:
            die(f"no frozen DREGON cruise support carries recording {recording!r}")
        support = found[0]
        clip = RE.load_window(
            support.window,
            dataset=GT.DATASET["dregon"],
            version=None,
            channels=None,
            rps_key=GT.RAW_RPS_KEY["dregon"],
        )
        sr = int(clip.sr)
        real = np.asarray(clip.audio, dtype=np.float64)[:n_mics]
        reference = np.atleast_2d(np.asarray(clip.rps, dtype=np.float64))
        rsupport = RE.regime_support(
            support.window,
            reference,
            regime=support.regime,
            min_rps=support.min_rps,
            max_rps=support.max_rps,
            sr=sr,
        )
        mics = list(range(min(int(n_mics), int(real.shape[0]))))
        row: dict[str, Any] = dict(support=support.as_dict(), arms={})
        reference_render = RD.render_noise(fit, reference, n_mics=len(mics), seed=int(seed))[
            : len(mics)
        ]
        larm = rs._arm_for_recording(legacy, support)
        for spec in ARMS:
            t0 = time.time()
            if spec.kind == "real":
                audio = real
            elif spec.kind == "legacy":
                audio = larm.render(
                    reference, regime=support.regime, n_mics=len(mics), seed=int(seed)
                )[: len(mics)]
            else:
                audio = render_phase(
                    fit,
                    reference,
                    sr=sr,
                    n_mics=len(mics),
                    seed=int(seed),
                    order_lock=spec.order_lock,
                    mic_lock=spec.mic_lock,
                    zero_gamma_k=spec.zero_gamma_k,
                    coherence_k_half=spec.coherence_k_half,
                )[: len(mics)]
            entry: dict[str, Any] = dict(
                tag=spec.tag,
                label=spec.label,
                kind=spec.kind,
                render_seconds=float(time.time() - t0),
                band_level_db_mic0=float(
                    10.0
                    * np.log10(
                        max(
                            float(
                                np.mean(
                                    np.abs(np.fft.rfft(audio[0] * np.hanning(audio.shape[-1]))) ** 2
                                )
                            ),
                            1e-300,
                        )
                    )
                ),
                **coherence_stats(audio, reference, sr=sr),
            )
            if spec.name == "v2":
                entry["reproduces_render_noise"] = dict(
                    max_abs_diff=float(np.abs(audio - reference_render).max()),
                    max_abs_render=float(np.abs(reference_render).max()),
                )
            if probe_obj is not None:
                pit = probe_obj.tracker.pit(
                    audio,
                    reference,
                    mics=mics,
                    expected_samples=int(real.shape[-1]),
                    support=rsupport,
                )
                entry["pit_mae"] = float(pit["mae"])
                entry["pit_per_mic"] = [float(v) for v in pit["per_mic"]]
            row["arms"][spec.name] = entry
            print(
                f"[{recording}] {spec.name} ({spec.tag}): "
                f"order_coh={entry['inter_order_mean']:.3f} "
                f"msc_k1={entry['inter_mic_msc'].get('k1'):.3f} "
                f"pit={entry.get('pit_mae')}",
                flush=True,
            )
        payload["supports"][support.key] = row
    Path(out).mkdir(parents=True, exist_ok=True)
    return payload


def findings(payload: dict[str, Any], *, job: str | None) -> str:
    rows = list(payload["supports"].values())
    recs = [str(r["support"]["recording"]).split("_")[0] for r in rows]
    arms = [a["name"] for a in payload["protocol"]["arms"]]
    tags = {a["name"]: a["tag"] for a in payload["protocol"]["arms"]}
    labels = {a["name"]: a["label"] for a in payload["protocol"]["arms"]}
    o = ["# DREGON round 3: the coherence probe", ""]
    o.append(
        "Comb POWER is held at the fitted `M` in every synthetic arm — "
        "`render_phase` changes only how the line phases are drawn — so a PIT "
        "difference here is a statistics difference and nothing else. The `v2` arm is "
        "`render_phase` with every option off and reproduces "
        "`noise_model.render.render_noise` to "
        + ", ".join(
            f"{r['arms']['v2']['reproduces_render_noise']['max_abs_diff']:.1e}" for r in rows
        )
        + " on peaks of "
        + ", ".join(
            f"{r['arms']['v2']['reproduces_render_noise']['max_abs_render']:.3f}" for r in rows
        )
        + f"; seed {payload['protocol']['seed']}, {payload['protocol']['n_mics']} mics"
        + (f", probe job `{job}`." if job else ".")
    )
    o.append("")
    o.append("| arm | tag | what it is |")
    o.append("|---|---|---|")
    for n in arms:
        o.append(f"| `{n}` | {tags[n]} | {labels[n]} |")
    o.append("")
    o.append("## PIT MAE (rev/s) and the two coherence statistics")
    o.append("")
    o.append(
        "| arm | tag | "
        + " | ".join(recs)
        + " | mean | inter-order coh (mean k=1..6) | inter-mic MSC k=1 | k=2 |"
    )
    o.append("|---|---|" + "---:|" * (len(recs) + 4))
    for n in arms:
        pits = [r["arms"][n].get("pit_mae") for r in rows]
        ok = [p for p in pits if p is not None]
        coh = float(np.mean([r["arms"][n]["inter_order_mean"] for r in rows]))
        m1 = float(np.mean([r["arms"][n]["inter_mic_msc"]["k1"] for r in rows]))
        m2 = float(np.mean([r["arms"][n]["inter_mic_msc"]["k2"] for r in rows]))
        o.append(
            f"| `{n}` | {tags[n]} | "
            + " | ".join("—" if p is None else f"{p:.3f}" for p in pits)
            + f" | {(np.mean(ok) if ok else float('nan')):.3f} | {coh:.3f} | {m1:.3f} | {m2:.3f} |"
        )
    o.append("")
    o.append(
        "Inter-order phase coherence is `|<z_k z_{k+1}^* / |.|>|` over the 4 s window on "
        f"the label's own carriers (time-domain demodulation, +-{DEMOD_BW_HZ:g} Hz), "
        "averaged over the four rotors, eight microphones and the five adjacent order "
        "pairs k=1..6 — it measures how STEADY the cross-order relative phase is. "
        "Inter-microphone MSC is the magnitude-squared coherence of the same demodulated "
        "envelope over the 28 microphone pairs. Both are 1 for a perfectly steady, "
        "spatially coherent line and fall towards 0 as the floor takes over the band."
    )
    o.append("")
    o.append("## Per-window detail")
    o.append("")
    o.append("| window | arm | PIT | inter-order coh | MSC k=1 | MSC k=2 |")
    o.append("|---|---|---:|---:|---:|---:|")
    for r, rec in zip(rows, recs, strict=True):
        for n in arms:
            e = r["arms"][n]
            pit = e.get("pit_mae")
            o.append(
                f"| `{rec}` | `{n}` | "
                + ("—" if pit is None else f"{pit:.3f}")
                + f" | {e['inter_order_mean']:.3f} | {e['inter_mic_msc']['k1']:.3f} | "
                f"{e['inter_mic_msc']['k2']:.3f} |"
            )
    o.append("")
    scorer = payload["protocol"].get("scorer") or {}
    o.append(
        "Probe `"
        + str(scorer.get("experiment", "—"))
        + "/"
        + str(scorer.get("ckpt", "—"))
        + "`, sha256 `"
        + str(scorer.get("sha256", "—"))
        + f"`. Record `{payload['git']}`"
        + (f", job `{job}`." if job else ".")
    )
    o.append("")
    return "\n".join(o)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", type=Path, default=FIT_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--n-mics", type=int, default=8)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--job", type=str, default=None)
    ap.add_argument("--recordings", default=",".join(RECORDINGS))
    args = ap.parse_args(argv)
    payload = run(
        fit_path=args.fit,
        out=args.out,
        seed=int(args.seed),
        probe=bool(args.probe),
        n_mics=int(args.n_mics),
        recordings=tuple(r for r in str(args.recordings).split(",") if r),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "coherence.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    (out / "coherence.md").write_text(findings(payload, job=args.job))
    print(f"wrote {out / 'coherence.json'}")
    print(f"wrote {out / 'coherence.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
