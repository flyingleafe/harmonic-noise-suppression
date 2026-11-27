#!/usr/bin/env python3
"""Sealed synthetic close-rotor pilot; run calibration, then test on uni-cpu.

All methods get the same blind acoustic proposal context. Null trials search
all the same seed paths as positives; one independent recording is one test.
The published peak picker is unmodified. The AVQ joint spectral scorer and
pi_kalman_refine(pair_mode='joint') are existing code, with local adapters to
turn their outputs into ranked split decisions. No true speed, gap, drift,
active count, or nuisance rate is used by any fitting/seed adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import __version__ as scipy_version
from scipy.signal import butter, lfilter, sosfilt
from scipy.stats import beta

from experiments.static_rig import close_pair, joint_speeds
from tracking import phase_increment_tracker, steady_tachometry

SCHEMA = "close-pair-controls/1"
BASELINE_PIN = "497ea2c5828a6cac19dc23ceabe3e1c4340725265ad99c06bc0218108797cd10"
PINNED_PEAK_SHA256 = "775ea3db11988faee47a0d14f0bcef94b985704edcfb75d8cf7833680c07796d"
FS = 16_000
ALPHA = 0.01
SEED_ROOTS = {"calibration": 2186661091, "test": 4026441169}
METHODS = ("phase", "peak", "two_tone", "joint_spectral")
NULLS = {"single": 230, "am_sidebands": 230, "exact_tie": 160, "rotor_off": 100}
DURATIONS = (2.5, 4.0, 8.0)
LEVELS = (-22.0, -14.0, -7.0, 0.0)
SNRS = (-8.0, 2.0, 12.0)
GAPS = (0.04, 0.08, 0.15, 0.25, 0.4, 0.65, 1.0, 1.4)
SEPARATED_GAPS = (1.8, 2.6, 3.4)
SEED_FRACTION = 0.48
TRACK_INIT_GAP = 0.75
FACTS = {"blade_count": 2, "relative_motion": "fixed"}


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_bytes(obj):
    return (json.dumps(obj, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()


def _write_json(path, obj):
    with path.open("xb") as f:
        f.write(_json_bytes(obj))


def _code_receipt():
    root = Path(steady_tachometry.__file__).resolve().parent
    modules = (close_pair, steady_tachometry, phase_increment_tracker, joint_speeds)
    hashes = {Path(__file__).name: _sha(__file__)}
    for module in modules:
        path = module.__file__
        assert path is not None
        hashes[Path(path).name] = _sha(path)
    return hashes | {name: _sha(root / name) for name in ("dsp.py", "tach_corroboration.py")}


def _plan(split):
    """Factorial stationary losses; independent seeds for every row and arm."""
    root = SEED_ROOTS[split]
    trial = 0
    for kind, n in NULLS.items():
        for _ in range(n):
            yield {"split": split, "root": root, "trial": trial, "kind": kind}
            trial += 1
    if split == "calibration":
        return
    for gap, db, snr, duration, rep in itertools.product(GAPS, LEVELS, SNRS, DURATIONS, range(2)):
        yield {
            "split": split,
            "root": root,
            "trial": trial,
            "kind": "static_close",
            "gap": gap,
            "relative_db": db,
            "snr_db": snr,
            "duration_s": duration,
            "rep": rep,
        }
        trial += 1
    for gap, db, snr, duration in itertools.product(SEPARATED_GAPS, LEVELS, SNRS, DURATIONS):
        yield {
            "split": split,
            "root": root,
            "trial": trial,
            "kind": "separated",
            "gap": gap,
            "relative_db": db,
            "snr_db": snr,
            "duration_s": duration,
        }
        trial += 1
    for kind in ("slow_drift", "crossing", "weak_fourth", "nuisance_pair", "center_tie_crossing"):
        for _ in range(100):
            yield {"split": split, "root": root, "trial": trial, "kind": kind}
            trial += 1


def _standard(x):
    return x / max(float(np.sqrt(np.mean(x * x))), 1e-20)


def _ou(rng, n, fs, hz, shape=()):
    a = math.exp(-2 * math.pi * hz / fs)
    tail = max(8, math.ceil(4 * fs / (2 * math.pi * hz)))
    w = rng.standard_normal((*shape, n + tail))
    return np.asarray(lfilter([math.sqrt(1 - a * a)], [1, -a], w, axis=-1))[..., tail:]


def _over_time(rng, n, fs, hz, shape=()):
    # Slow envelopes need no native-rate filtering. Interpolation is shared
    # across mic views for each source; speakers/mics do not share a clock fit.
    ef = 100
    m = math.ceil(n * ef / fs) + 1
    w = _ou(rng, m, ef, hz, shape)
    a, b = np.arange(n) / fs, np.arange(m) / ef
    return np.apply_along_axis(lambda r: np.interp(a, b, r), -1, w)


def _shaft(rng, t, rate, drift, diffusion):
    # Stationary OU frequency jitter with long-run phase variance D * T.
    lam = 2 * math.pi * 12.0
    if_hz = _over_time(rng, t.size, FS, 12.0) * math.sqrt(diffusion * lam / 2)
    phase_noise = np.cumsum(if_hz) / FS
    tc = t.size // 2
    return rate * t + 0.5 * drift * t * t + phase_noise - phase_noise[tc]


def _rotor(rng, t, microphones, rate, phase, common_am, *, sideband=None, exact_tie=False):
    # Even blade-passing harmonics dominate with real odd-order asymmetry;
    # profiles, missing orders, gains, path delays and image vary by source.
    k = np.arange(1, min(50, int(0.43 * FS / rate)) + 1)
    azimuth = rng.uniform(0, 2 * np.pi)
    rotor_xyz = np.array(
        [
            rng.uniform(0.07, 0.28) * np.cos(azimuth),
            rng.uniform(0.07, 0.28) * np.sin(azimuth),
            rng.uniform(0.2, 1.8),
        ]
    )
    d = np.linalg.norm(microphones - rotor_xyz, axis=1)
    image = d * rng.uniform(1.05, 1.30) + rng.uniform(0.02, 0.65, size=d.size)
    line = (2.0 / np.maximum(2, k)) ** rng.uniform(0.7, 1.35)
    line *= np.where(k % 2, rng.uniform(0.04, 0.22, len(k)), 1.0)
    line *= np.exp(rng.normal(0, 0.55, len(k)))
    line[k == 2] = max(float(line.max()), 1.0)
    line[(k > 7) & (rng.random(len(k)) < rng.uniform(0.05, 0.45))] = 0
    # Add a path image (frequency-dependent mic gains) without modifying the
    # rate clock. Occasional pickup notch from constructive/destructive sound.
    theta = -2j * np.pi * (k[:, None] * rate) / 343.0
    steering = (np.exp(theta * d) + rng.uniform(0.12, 0.72) * np.exp(theta * image)) / d
    steering *= np.exp(rng.normal(0, 0.23, steering.shape))
    line_phase = rng.uniform(0, 1, len(k))
    am_scale = rng.uniform(0.05, 0.16)
    am_hz = rng.uniform(0.25, 0.65)
    am_phase = rng.uniform(0, 2 * np.pi, len(k))
    locked_am = np.exp(np.clip(common_am, -1.3, 1.3)) if exact_tie else None
    z = np.zeros((len(microphones), len(t)))
    for a in range(0, len(k), 8):
        sl = slice(a, a + 8)
        ks = k[sl, None]
        carrier = np.exp(2j * np.pi * (ks * phase + line_phase[sl, None]))
        if sideband is not None:
            modulation_hz, proportional, depth, asymmetry = sideband
            hz = modulation_hz * (ks if proportional else np.ones_like(ks))
            u = 2 * np.pi * hz * t + am_phase[sl, None]
            # Real AM has symmetric satellites; complex AM+PM also permits
            # asymmetric satellites around a rotor tone.
            carrier *= 1 + depth * (np.cos(u) + 1j * asymmetry * np.sin(u))
        if locked_am is not None:
            carrier *= locked_am
        else:
            env = _over_time(rng, len(t), FS, am_hz, (len(ks),))
            carrier *= np.exp(np.clip(common_am + am_scale * env, -1.3, 1.3))
        z += ((steering[sl] * line[sl, None]).T @ carrier).real
    return _standard(z)


def _voiced_interference(rng, t, c):
    """Synthetic off-motor voiced-like pulse train + formants; not real speech."""
    n = len(t)
    f0 = rng.uniform(88, 210) + rng.uniform(4, 33) * np.sin(2 * np.pi * rng.uniform(0.3, 1.6) * t)
    f0 += rng.uniform(1, 6) * np.sin(2 * np.pi * rng.uniform(3, 7) * t + rng.uniform(0, 6.3))
    ph = np.cumsum(f0) / FS
    saw = (ph % 1 > 0.09).astype(float) - (ph % 1 < 0.8).astype(float)
    sig = np.zeros(n)
    for lo, hi in ((230, 680), (750, 1550), (1700, 3100)):
        f = rng.uniform(lo, hi)
        q = rng.uniform(0.05, 0.21) * f
        sig += sosfilt(butter(2, (f - q, f + q), fs=FS, btype="bandpass", output="sos"), saw)
    syllables = np.maximum(0.0, np.sin(2 * np.pi * rng.uniform(1.2, 4.8) * t + rng.uniform(0, 6.3)))
    sig *= 0.20 + 0.80 * syllables
    delay = rng.integers(0, 17, size=c)
    images = rng.integers(10, 165, size=c)
    return _standard(
        np.stack(
            [np.roll(sig, d) + 0.25 * np.roll(sig, i) for d, i in zip(delay, images, strict=True)]
        )
    )


def _render(spec):
    rng = np.random.default_rng(np.random.SeedSequence((spec["root"], spec["trial"])))
    kind = spec["kind"]
    duration = float(spec.get("duration_s", rng.choice(DURATIONS)))
    n = round(duration * FS)
    t = (np.arange(n) - n / 2) / FS
    c = int(rng.choice((1, 4, 8), p=(0.16, 0.48, 0.36)))
    microphones = np.zeros((c, 3))
    microphones[:, :2] = rng.uniform(-0.16, 0.16, (c, 2))
    center = float(rng.uniform(58.0, 125.0))
    gap = float(spec.get("gap", rng.choice(GAPS)))
    rel = float(spec.get("relative_db", rng.choice(LEVELS)))
    if kind == "weak_fourth":
        rel = float(rng.choice((-26.0, -22.0, -18.0, -14.0, -10.0)))
    snr_db = float(spec.get("snr_db", rng.choice(SNRS)))
    diffusion = float(np.exp(rng.uniform(np.log(0.3e-4), np.log(2.5e-4))))
    primary_drift = float(rng.normal(0, 0.004))
    d = np.full(2, primary_drift)
    if kind in ("slow_drift", "crossing", "center_tie_crossing"):
        d += rng.uniform(0.03, 0.23) * rng.choice((-1, 1))
    if kind in ("crossing", "center_tie_crossing"):
        crossing_s = (
            0.0
            if kind == "center_tie_crossing"
            else duration * rng.uniform(0.12, 0.38) * rng.choice((-1, 1))
        )
        dd = rng.uniform(0.13, 0.60) if kind == "center_tie_crossing" else -gap / crossing_s
        d += np.array((-dd / 2, dd / 2))
        gap = abs(dd * crossing_s)
    elif kind == "slow_drift":
        dd = min(gap / duration, 0.30) * rng.choice((-1, 1))
        d += np.array((-dd / 2, dd / 2))
    elif kind in NULLS:
        gap = 0.0
    rates = np.array([center - gap / 2, center + gap / 2])
    if kind in NULLS and kind != "exact_tie":
        rates = rates[:1] if kind != "rotor_off" else rates[:0]
    t_cross = float(-gap / (d[1] - d[0])) if d[1] != d[0] else None
    common_am = rng.uniform(0.19, 0.32) * _over_time(rng, n, FS, rng.uniform(0.3, 0.5))
    sideband_options = None
    if kind == "am_sidebands":
        proportional = bool(rng.integers(0, 2))
        mod_hz = rng.uniform(0.05, 0.42) if proportional else rng.uniform(0.35, 5.0)
        sideband_options = (
            float(mod_hz),
            proportional,
            float(rng.uniform(0.16, 0.54)),
            float(rng.uniform(-1, 1)),
        )
    rotors = []
    if len(rates):
        p = _shaft(rng, t, rates[0], d[0], diffusion)
        exact_tie = kind == "exact_tie"
        rotors.append(
            _rotor(
                rng,
                t,
                microphones,
                rates[0],
                p,
                common_am,
                sideband=sideband_options,
                exact_tie=exact_tie,
            )
        )
        if len(rates) == 2:
            # At the exact tie both sources share the time-dependent
            # coefficient and shaft phase. Only their fixed gains differ.
            if exact_tie:
                rotors.append(_rotor(rng, t, microphones, rates[1], p, common_am, exact_tie=True))
            else:
                q = _shaft(rng, t, rates[1], d[1], diffusion)
                other_am = rng.uniform(0.19, 0.32) * _over_time(rng, n, FS, rng.uniform(0.3, 0.5))
                rotors.append(_rotor(rng, t, microphones, rates[1], q, other_am))
    level = 10 ** (rel / 20)
    weights = (level, 1.0) if len(rotors) == 2 and rng.random() < 0.5 else (1.0, level)
    signal = sum((w * r for w, r in zip(weights, rotors)), start=np.zeros((c, n)))
    nuisance_rates = []
    other = np.zeros((c, n))
    if kind == "weak_fourth":
        nuisance_rates = [
            float(center - rng.uniform(6.2, 9.5)),
            float(center + rng.uniform(7.5, 13.0)),
        ]
        for r in nuisance_rates:
            p = _shaft(rng, t, r, 0.0, diffusion)
            a = rng.uniform(0.18, 0.28) * _over_time(rng, n, FS, 0.5)
            other += rng.uniform(0.75, 1.35) * _rotor(rng, t, microphones, r, p, a)
    if kind in ("nuisance_pair", "rotor_off") or rng.random() < 0.24:
        other += 10 ** (rng.uniform(-15.0, 0.0) / 20) * _voiced_interference(rng, t, c)
        # Include non-shaft and intermodulation tones; nuisance rates always
        # absent from fit inputs except when the acoustic seed finder sees them.
        f1 = rng.uniform(90, 1900)
        f2 = 2 * FS * 0.001 + f1 + rng.uniform(-12, 12)
        if len(rates) == 2 and kind != "exact_tie":
            f2 = 2 * rates[0] + rates[1]
        for f in (f1, f2):
            other += rng.uniform(0.025, 0.15) * np.sin(
                2 * np.pi * f * t + rng.uniform(0, 6.28, (c, 1))
            )
    background = _standard(
        lfilter([1], [1, -rng.uniform(0.5, 0.97)], rng.standard_normal((c, n)), axis=-1)
    )
    background *= np.exp(0.16 * _over_time(rng, n, FS, 0.8, (c,)))
    noise = _standard(0.7 * background + 0.35 * rng.standard_normal((c, n)))
    noise *= max(np.sqrt(np.mean(signal * signal)), 0.22) * 10 ** (-snr_db / 20)
    x = signal + other + noise
    x *= 0.56 / max(0.56, float(np.abs(x).max()))  # native linear scale; no clipping
    x = x.astype(np.float32)
    return x, {
        **spec,
        "duration_s": duration,
        "channels": c,
        "snr_db": snr_db,
        "native_pcm_sha256": hashlib.sha256(memoryview(x)).hexdigest(),
        "relative_db": rel,
        "component_weights": list(weights[: len(rates)]),
        "truth_rps": rates.tolist(),
        "truth_gap_rps": gap if len(rates) == 2 else None,
        "truth_drift_rps_per_s": d[: len(rates)].tolist(),
        "truth_crossing_s_from_center": t_cross,
        "nuisance_rps": nuisance_rates,
        "am_sideband": sideband_options,
        "phase_diffusion_rev2_per_s": diffusion,
    }


def _center_window(result, t):
    options = [w for w in result["windows"] if w["start_s"] <= t <= w["end_s"]]
    return min(options, key=lambda w: abs(t - w["center_s"]), default={})


def _contexts(x):
    """Seed on prefix only, before prediction; no source-level multiplicity."""
    prefix = x[:, : int(x.shape[-1] * SEED_FRACTION)]
    # 1.5 s is the release's minimum. An unconfirmed full-prefix comb
    # proposal is allowed below that or when no public family survives.
    private = steady_tachometry.estimate(prefix, FS, FACTS, candidate_only=True)
    public = steady_tachometry.corroborate(private, FACTS)
    center = prefix.shape[-1] / (2 * FS)
    window = _center_window(public, center)
    committed = tuple(window.get("rates_rev_s", ()))
    provisional = tuple(_center_window(private, center).get("rates_rev_s", ()))
    source = "published_prefix" if committed else "private_prefix"
    if not committed and not provisional:
        phys = steady_tachometry._physics(FACTS)
        p = steady_tachometry._peaks(prefix, FS)
        combs, _ = steady_tachometry._select(p, phys, np.empty(0))
        provisional = tuple(q.rate for q in combs)
        source = "unchecked_short_prefix"
    proposals = committed or provisional
    # Try each family alone; keep already-observed neighbors separate, and
    # also assess already-resolved adjacent pairs without excluding a member.
    contexts = [(tuple([r]), tuple(s for s in committed if s != r)) for r in proposals]
    contexts += [
        ((a, b), tuple(s for s in committed if s not in (a, b)))
        for a, b in zip(committed, committed[1:])
        if 0 < b - a <= 4.25
    ]
    return source, contexts, committed


def _none(reason):
    return {"rates": [], "score": None, "eligible": False, "reason": reason}


def _finite_score(result):
    score = result.get("score")
    return float(score) if score is not None and math.isfinite(score) else None


def _winner(results):
    usable = [
        (s, i)
        for i, row in enumerate(results)
        if row.get("eligible")
        and len(row.get("rates", ())) == 2
        and (s := _finite_score(row)) is not None
    ]
    if usable:
        return results[max(usable, key=lambda a: (a[0], -a[1]))[1]]
    scored = [(s, r) for r in results if (s := _finite_score(r)) is not None]
    return (
        max(scored, key=lambda item: item[0])[1]
        if scored
        else (results[0] if results else _none("no_acoustic_seed"))
    )


def _radius(seed):
    return 2.8 if len(seed) == 1 else max(1.7, (seed[1] - seed[0]) / 2 + 1.05)


def _local(rates, seed, other):
    center = float(np.mean(seed))
    return (
        0 < rates[1] - rates[0] <= 5.0
        and max(abs(r - center) for r in rates) <= _radius(seed)
        and all(abs(r - s) > 0.12 for r in rates for s in other)
    )


def _peak(x, contexts):
    """Pinned estimator at its native gate plus a rank from its archived evidence."""
    res = steady_tachometry.estimate(x, FS, FACTS)
    w = _center_window(res, x.shape[-1] / (2 * FS))
    speeds = w.get("rates_rev_s", ())
    checks = w.get("quality", {}).get("acoustic_crosscheck", ())
    scored = []
    for i, (r, support) in enumerate(zip(speeds, w.get("order_support", ()), strict=True)):
        close = [
            k["primary"]
            for k in checks
            if k["primary"] is not None and abs(k["primary"]["rate_rev_s"] - r) < 0.25
        ]
        strength = max((m["score"] for m in close), default=float(len(support)))
        scored.append((float(r), float(strength), w["uncertainty_rev_s"][i]))
    pairs = []
    for (a, sa, ua), (b, sb, ub) in itertools.combinations(scored, 2):
        if any(_local((a, b), seed, other) for seed, other in contexts):
            pairs.append(
                {
                    "rates": [a, b],
                    "score": min(sa, sb),
                    "eligible": True,
                    "reason": "release_corroborated_pair",
                    "uncertainty_rps": [ua, ub],
                }
            )
    best = _winner(pairs)
    best = {
        **best,
        "release_rates_rps": list(speeds),
        "release_window_s": [w["start_s"], w["end_s"]] if w else [],
    }
    return best, pairs


def _two_tone(x, seed, other):
    """Public PI Kalman two-tone branch; never count an unmeasured init as split."""
    center = float(np.mean(seed))
    guess = seed if len(seed) == 2 else (center - TRACK_INIT_GAP / 2, center + TRACK_INIT_GAP / 2)
    ft = np.arange(0, x.shape[-1] / FS, 0.032)
    init = np.broadcast_to(np.asarray((*guess, *other))[:, None], (2 + len(other), len(ft)))
    tracks, diag = phase_increment_tracker.pi_kalman_refine(
        x, init, ft, FS, pair_mode="joint", threads=1
    )
    middle = x.shape[-1] / (2 * FS)
    rates = sorted(float(np.interp(middle, ft, t)) for t in tracks[:2])
    lock = [p for it in diag["pairs"] for p in it if p["pair"] in ([0, 1], [1, 0])]
    n_locked = sum(p.get("n_windows_locked", 0) for p in lock)
    n_windows = sum(p.get("n_windows", 0) for p in lock)
    separated_init = len(seed) == 2 and abs(seed[1] - seed[0]) >= 1.5
    increment_counts = [
        max((sum(r.get("n_meas_k", {}).values()) for r in d["iters"]), default=0)
        for d in diag["rotors"][:2]
    ]
    if not _local(rates, seed, other):
        out = _none("legacy_outside_dispute")
    elif n_locked:
        out = {
            "rates": rates,
            "score": n_locked / (n_windows + 8.0),
            "eligible": True,
            "reason": "legacy_two_tone_observed",
        }
    elif separated_init and min(increment_counts) > 0:
        # The joint pipeline intentionally uses isolated increment evidence for
        # pairs outside its 1.5 rev/s coupling radius, not fake joint locks.
        out = {
            "rates": rates,
            "score": 1.0 + min(increment_counts) / (min(increment_counts) + 250.0),
            "eligible": True,
            "reason": "legacy_separated_measured",
        }
    else:
        out = _none("legacy_no_two_tone_observation")
    return {
        **out,
        "init_rps": list(guess),
        "joint_n_locked": n_locked,
        "joint_n_windows": n_windows,
        "legacy_independent_counts": increment_counts,
    }


def _joint_spectral(x, seed, other):
    """AVQ model's real enumerated 1-vs-2 evidence; no forced R=2 acceptance."""
    center = float(np.mean(seed))
    p = joint_speeds.JointParams(lo=max(15.0, center - _radius(seed)), hi=center + _radius(seed))
    w = joint_speeds.WindowEvidence(x, FS, p)
    ones = [m for m in w.modes(1) if all(abs(m["speeds"][0] - r) > 0.12 for r in other)]
    twos = [m for m in w.modes(2) if _local(m["speeds"], seed, other)]
    if not ones or not twos:
        return _none("avq_no_distinct_local_pair")
    # Score known side families under both hypotheses, with the library's
    # collision and unexplained-cell penalty, rather than pretend they are noise.
    e1 = max(w.joint([*m["speeds"], *other]) for m in ones)
    e2, rates = max((w.joint([*m["speeds"], *other]), m["speeds"]) for m in twos)
    return {
        "rates": rates,
        "score": float(e2 - e1),
        "eligible": True,
        "reason": "avq_joint_evidence",
    }


def _run_trial(spec):
    x, row = _render(spec)
    seed_source, contexts, committed = _contexts(x)
    phase, legacy, spectral = [], [], []
    for seed, other in contexts:
        context = {"seed_rps": seed, "other_rates": other}
        fit = close_pair.fit_pair(x, FS, seed, other_rates=other)
        if fit.get("eligible") and not _local(fit["rates"], seed, other):
            fit = {
                **fit,
                "eligible": False,
                "reason": "outside_local_dispute_or_duplicates_neighbor",
            }
        phase.append({**fit, **context})
        legacy.append({**_two_tone(x, seed, other), **context})
        spectral.append({**_joint_spectral(x, seed, other), **context})
    peak, peak_attempts = _peak(x, contexts)
    attempts = {
        "phase": phase,
        "peak": peak_attempts,
        "two_tone": legacy,
        "joint_spectral": spectral,
    }
    return {
        **row,
        "seed_source": seed_source,
        "prefix_public_rates_rps": committed,
        "attempts": attempts,
        "methods": {
            "phase": _winner(phase),
            "peak": peak,
            "two_tone": _winner(legacy),
            "joint_spectral": _winner(spectral),
        },
    }


def _threshold(rows, method):
    has = [
        s
        for r in rows
        if r["methods"][method].get("eligible")
        and (s := _finite_score(r["methods"][method])) is not None
        and s > 0
    ]
    s = sorted(has, reverse=True)
    allowed = math.floor(ALPHA * len(rows))
    cutoff = 0.0 if len(s) <= allowed else s[allowed]
    accepted = sum(v > cutoff for v in s)
    return {
        "score_gt": cutoff,
        "accept_all_scored": False,
        "reject_all": False,
        "calibration_accepts": accepted,
        "calibration_rate_ci95": _binomial_ci(accepted, len(rows)),
        "calibration_n_scored": len(s),
        "calibration_n": len(rows),
        "budget": allowed,
        "ties_at_cutoff": s.count(cutoff),
    }


def _accept(row, threshold):
    s = _finite_score(row)
    return bool(
        row.get("eligible")
        and s is not None
        and not threshold["reject_all"]
        and (threshold["accept_all_scored"] or s > threshold["score_gt"])
    )


def _binomial_ci(k, n):
    if not n:
        return None
    a = 0.025
    return [
        float(beta.ppf(a, k, n - k + 1)) if k else 0.0,
        float(beta.ppf(1 - a, k + 1, n - k)) if k < n else 1.0,
    ]


def _pctl(x):
    return (
        dict(
            zip(
                ("p50", "p90", "p95", "max"),
                map(float, np.quantile(x, (0.5, 0.90, 0.95, 1.0))),
                strict=True,
            )
        )
        if x
        else None
    )


def _measure(row, method, thr):
    r = row["methods"][method]
    accepted = _accept(r, thr)
    gap = row["truth_gap_rps"]
    errors, drift_errors = None, None
    matched = False
    interval = r.get("separation_interval_rps")
    has_interval = (
        isinstance(interval, (tuple, list))
        and len(interval) == 2
        and all(v is not None for v in interval)
    )
    cover = bool(interval[0] <= gap <= interval[1]) if has_interval and gap is not None else None
    if len(r.get("rates", ())) == 2 and len(row["truth_rps"]) == 2:
        errors = np.abs(np.asarray(r["rates"]) - row["truth_rps"]).tolist()
        # Stricter than a constant 0.2 rev/s: a coincident midpoint cannot
        # be credited with resolving both shafts in a very close trial.
        if gap and gap > 0:
            matched = bool(accepted and max(errors) <= min(0.20, 0.30 * gap))
        d = r.get("drift_rps_per_s")
        if isinstance(d, (tuple, list)) and len(d) == 2:
            drift_errors = np.abs(np.asarray(d) - row["truth_drift_rps_per_s"]).tolist()
    return {
        "accepted": accepted,
        "matched": matched,
        "rate_error_rps": errors,
        "abs_drift_error_rps_per_s": drift_errors,
        "interval_covers": cover,
    }


def _release_addon(row, phase_threshold):
    """Retain accepted rates verbatim; only augment a singly anchored dispute."""
    original = row["methods"]["peak"]["release_rates_rps"]
    candidates = []
    for r in row["attempts"]["phase"]:
        seed = r["seed_rps"]
        if len(seed) != 1 or not _accept(r, phase_threshold):
            continue
        anchors = [s for s in original if abs(s - seed[0]) <= 0.25]
        if len(anchors) != 1:
            continue
        anchor = anchors[0]
        observed, extra = sorted(r["rates"], key=lambda s: abs(anchor - s))
        if abs(observed - anchor) > 0.25 or min(abs(extra - s) for s in original) < 0.025:
            continue
        if any(abs(s - seed[0]) <= _radius(seed) for s in original if s != anchor):
            continue  # already resolved two rates in this neighborhood
        candidates.append((_finite_score(r), extra))
    chosen = max(candidates, default=(None, None))[1]
    augmented = sorted((*original, chosen)) if chosen is not None else original.copy()

    def matched(rates):
        gap = row["truth_gap_rps"]
        if not gap or gap <= 0 or len(row["truth_rps"]) != 2:
            return False
        tol = min(0.20, 0.30 * gap)
        a, b = row["truth_rps"]
        return any(
            abs(u - a) <= tol and abs(v - b) <= tol for u, v in itertools.combinations(rates, 2)
        )

    truth_full = sorted((*row["truth_rps"], *row["nuisance_rps"]))

    def matched_full(rates):
        return bool(
            len(truth_full) == 4
            and len(rates) == 4
            and max(abs(r - t) for r, t in zip(rates, truth_full, strict=True)) <= 0.20
            and matched(rates)
        )

    return {
        "native_rates_rps": original,
        "augmented_rates_rps": augmented,
        "added_rate_rps": chosen,
        "native_matched": matched(original),
        "augmented_matched": matched(augmented),
        "native_all_four_matched": matched_full(original),
        "augmented_all_four_matched": matched_full(augmented),
    }


def _addon_stats(rows):
    o = [r["operational"] for r in rows]
    native = sum(len(r["native_rates_rps"]) >= 2 for r in o)
    augmented = sum(len(r["augmented_rates_rps"]) >= 2 for r in o)
    return {
        "n": len(o),
        "native_two_or_more": native,
        "augmented_two_or_more": augmented,
        "native_two_or_more_ci95": _binomial_ci(native, len(o)),
        "augmented_two_or_more_ci95": _binomial_ci(augmented, len(o)),
        "added_rate": sum(r["added_rate_rps"] is not None for r in o),
        "native_matched": sum(r["native_matched"] for r in o),
        "augmented_matched": sum(r["augmented_matched"] for r in o),
        "native_all_four_matched": sum(r["native_all_four_matched"] for r in o),
        "augmented_all_four_matched": sum(r["augmented_all_four_matched"] for r in o),
        "lost_release_rates": sum(
            not set(r["native_rates_rps"]) <= set(r["augmented_rates_rps"]) for r in o
        ),
    }


def _stats(rows, method):
    a = [r["assessment"][method] for r in rows]
    accepted = [x for x in a if x["accepted"]]
    matches = [x for x in accepted if x["matched"]]
    cov = [x["interval_covers"] for x in accepted if x["interval_covers"] is not None]
    all_cov = [x["interval_covers"] for x in a if x["interval_covers"] is not None]
    n = len(rows)
    return {
        "n": n,
        "accepted": len(accepted),
        "accepted_rate": len(accepted) / n,
        "matched": len(matches),
        "matched_rate": len(matches) / n,
        "accepted_rate_ci95": _binomial_ci(len(accepted), n),
        "matched_rate_ci95": _binomial_ci(len(matches), n),
        "accepted_rate_error_rps": _pctl(
            [v for x in accepted for v in (x["rate_error_rps"] or ())]
        ),
        "matched_rate_error_rps": _pctl([v for x in matches for v in (x["rate_error_rps"] or ())]),
        "accepted_abs_drift_error_rps_per_s": _pctl(
            [v for x in accepted for v in (x["abs_drift_error_rps_per_s"] or ())]
        ),
        "accepted_interval_n": len(cov),
        "accepted_interval_covered": sum(cov),
        "accepted_interval_coverage_ci95": _binomial_ci(sum(cov), len(cov)),
        "all_interval_n": len(all_cov),
        "all_interval_covered": sum(all_cov),
    }


def _paired_bootstrap(rows):
    rng = np.random.default_rng(0x4419C)
    a = np.array([[r["assessment"][m]["matched"] for m in METHODS] for r in rows], dtype=float)
    idx = rng.integers(0, len(rows), size=(6000, len(rows)))
    probs = a[idx].mean(axis=1)
    observed = a.mean(axis=0)
    return {
        m: {
            "delta": float(observed[0] - observed[j]),
            "paired_ci95": np.quantile(probs[:, 0] - probs[:, j], (0.025, 0.975)).tolist(),
        }
        for j, m in enumerate(METHODS[1:], start=1)
    } | {
        "best_each_resample": {
            "delta": float(observed[0] - max(observed[1:])),
            "paired_ci95": np.quantile(
                probs[:, 0] - probs[:, 1:].max(axis=1), (0.025, 0.975)
            ).tolist(),
        }
    }


def _matched_roc(rows, close, fpr):
    """Post-lock, descriptive comparison at the same held-out FPR envelope.

    No threshold from this ROC is ever used to accept a row or supplied to
    the real trial. Group score ties; don't select true positives inside ties.
    """
    out = {}
    nulls = [r for r in rows if r["kind"] in NULLS]
    budget = math.floor(fpr * len(nulls) + 1e-12)
    for m in METHODS:
        points = {}
        for r in (*nulls, *close):
            fit = r["methods"][m]
            score = _finite_score(fit)
            if not fit["eligible"] or score is None or score <= 0:
                continue
            f = points.setdefault(score, [0, 0])
            if r["kind"] in NULLS:
                f[0] += 1
            elif _measure(r, m, {"reject_all": False, "accept_all_scored": True})["matched"]:
                f[1] += 1
        fp = tp = 0
        for s in sorted(points, reverse=True):
            a, b = points[s]
            if a + fp > budget:
                break
            fp += a
            tp += b
        out[m] = {
            "fp": fp,
            "null_n": len(nulls),
            "matched_tp": tp,
            "close_n": len(close),
            "fpr": fp / len(nulls),
            "matched_recall": tp / len(close),
        }
    return out


def _summary(rows, threshold):
    groups = {"null_all": [r for r in rows if r["kind"] in NULLS]}
    for kind in sorted(set(r["kind"] for r in rows)):
        groups[kind] = [r for r in rows if r["kind"] == kind]
    close = groups["static_close"]
    marginals = {
        key: {
            str(v): {m: _stats([r for r in close if r[key] == v], m) for m in METHODS}
            for v in sorted(set(r[key] for r in close))
        }
        for key in ("truth_gap_rps", "relative_db", "snr_db", "duration_s", "channels")
    }
    limit = {}
    for duration, db, snr in itertools.product(DURATIONS, LEVELS, SNRS):
        key = f"{duration}s/{db}dB/{snr}dB"
        # Two independent replicates per cell. This is a descriptive 2/2
        # frontier, not evidence of a population >=90% recovery boundary.
        limit[key] = {
            m: next(
                (
                    g
                    for g in GAPS
                    if sum(
                        r["assessment"][m]["matched"]
                        for r in close
                        if (r["truth_gap_rps"], r["relative_db"], r["snr_db"], r["duration_s"])
                        == (g, db, snr, duration)
                    )
                    == 2
                ),
                None,
            )
            for m in METHODS
        }
    by_group = {k: {m: _stats(v, m) for m in METHODS} for k, v in groups.items()}
    comparisons = _paired_bootstrap(close)
    at_target = _matched_roc(rows, close, ALPHA)
    matched_fpr = min(ALPHA, by_group["null_all"]["phase"]["accepted_rate"])
    operational = {k: _addon_stats(v) for k, v in groups.items()}
    separated = operational["separated"]
    no_loss = (
        separated["lost_release_rates"] == 0
        and separated["augmented_matched"] >= separated["native_matched"]
    )
    return {
        "schema": SCHEMA,
        "threshold_sha256": threshold,
        "by_group": by_group,
        "static_close_marginals": marginals,
        "smallest_gap_at_2_of_2_matched_rps_descriptive": limit,
        "static_close_paired_comparison_at_locked_thresholds": comparisons,
        "heldout_matched_roc_at_common_1pct_fpr_ceiling": at_target,
        "heldout_matched_roc_at_phase_observed_fpr_ceiling": _matched_roc(rows, close, matched_fpr),
        "operational_native_release_and_retaining_addon": operational,
        "synthetic_gates": {
            "heldout_null_point_rate_at_most_1pct": by_group["null_all"]["phase"]["accepted_rate"]
            <= ALPHA,
            "close_match_advantage_at_locked_thresholds_paired_ci_lower_above_zero": comparisons[
                "best_each_resample"
            ]["paired_ci95"][0]
            > 0,
            "close_match_advantage_at_common_1pct_fpr_ceiling": at_target["phase"]["matched_tp"]
            > max(at_target[method]["matched_tp"] for method in METHODS[1:]),
            "no_separated_loss_in_release_retaining_policy": no_loss,
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("split", choices=tuple(SEED_ROOTS))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threshold", type=Path)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    plan = list(_plan(args.split))
    sealed_test_hash = hashlib.sha256(_json_bytes(list(_plan("test")))).hexdigest()
    hashes = _code_receipt()
    if hashes["steady_tachometry.py"] != PINNED_PEAK_SHA256:
        ap.error("peak-based baseline source does not match the pinned release gate")
    source = {
        "schema": SCHEMA,
        "split": args.split,
        "jobs": [os.environ.get("SLURM_JOB_ID")],
        "source_snapshot_sha256": os.environ.get("CLOSE_PAIR_SNAPSHOT_SHA256"),
        "worker_host": platform.node(),
        "n_workers": args.workers,
        "baseline_companion": BASELINE_PIN,
        "source_sha256": hashes,
        "sample_rate_hz": FS,
        "seed_prefix_fraction": SEED_FRACTION,
        "source_blades": 2,
        "known_installed_count": None,
        "native_peak_recipe": [steady_tachometry.RECIPE, steady_tachometry.ACCEPTANCE_RECIPE],
        "legacy_joint_init_single_seed_gap_rps": TRACK_INIT_GAP,
        "seed_root": SEED_ROOTS[args.split],
        "root_to_seed": "numpy SeedSequence((root, trial))",
        "sealed_test_plan_sha256": sealed_test_hash,
        "primary_endpoint": "static_close matched two distinct rates; max sorted error <= min(.20 rev/s, .30 * true gap)",
        "uncertainty_endpoint": "nominal center gap in support interval of top fit; report accepted and all-trial winner denominators",
        "plan_sha256": hashlib.sha256(_json_bytes(plan)).hexdigest(),
        "n": len(plan),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy_version,
    }
    threshold, locked_hash = None, None
    if args.split == "test":
        if args.threshold is None:
            ap.error("test requires the frozen --threshold calibration JSON")
        locked_hash = _sha(args.threshold)
        threshold = json.loads(args.threshold.read_text())
        if threshold["source_sha256"] != hashes or threshold["baseline_companion"] != BASELINE_PIN:
            ap.error("test source snapshot or baseline differs from the locked calibration")
        if (
            threshold["seed_root"] != SEED_ROOTS["calibration"]
            or threshold["schema"] != SCHEMA
            or threshold["sealed_test_plan_sha256"] != sealed_test_hash
        ):
            ap.error("invalid sealed split")
        source["threshold_sha256"] = locked_hash
    elif args.threshold is not None:
        ap.error("calibration does not take --threshold")
    args.out.mkdir(exist_ok=False, parents=True)
    _write_json(args.out / "plan.json", {**source, "trials": plan})
    if threshold is not None:
        _write_json(args.out / "threshold_used.json", threshold)
    rows = []
    rows_path = args.out / "rows.jsonl"
    # Spawn: legacy Torch FFT thread-pool configuration must be private to a
    # worker. Every task carries its own SeedSequence and is order invariant.
    started = time.perf_counter()
    with (
        rows_path.open("xb") as f,
        ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool,
    ):
        for r in pool.map(_run_trial, plan, chunksize=2):
            if threshold is not None:
                r["assessment"] = {m: _measure(r, m, threshold["thresholds"][m]) for m in METHODS}
                r["operational"] = _release_addon(r, threshold["thresholds"]["phase"])
            f.write(_json_bytes(r))
            rows.append(r)
            if len(rows) % 24 == 0:
                f.flush()
                print(f"{args.split}: {len(rows)}/{len(plan)}", flush=True)
    receipt = {
        **source,
        "raw_rows_sha256": _sha(rows_path),
        "fit_wall_seconds": round(time.perf_counter() - started, 3),
    }
    if args.split == "calibration":
        threshold = {
            **receipt,
            "target_fpr": ALPHA,
            "thresholds": {m: _threshold(rows, m) for m in METHODS},
        }
        _write_json(args.out / "threshold.json", threshold)
        receipt["threshold_sha256"] = _sha(args.out / "threshold.json")
        print(json.dumps(threshold["thresholds"], sort_keys=True), flush=True)
    else:
        _write_json(args.out / "summary.json", _summary(rows, locked_hash))
        receipt["summary_sha256"] = _sha(args.out / "summary.json")
    _write_json(args.out / "receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
