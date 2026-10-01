"""Rotor tracks and trajectory-demodulated harmonic amplitudes of a fixed rig
whose rotor speeds DRIFT (AVQ, KU Leuven, DroneAudioSet, ...).

The full-record read (:mod:`.spectra`) needs constant speeds. Where the speeds
drift, each rotor is tracked in time with the in-tree blind seeder
(:func:`tracking.comb_seed.seed_from_gram`: comb-gram Viterbi ridges, peeled by
notching) on one channel, refined on all channels by coupled Vold–Kalman order
tracking (:func:`tracking.vk_tracking.vk_track`), and its harmonics are read
off the final VK envelopes: per rotor, order and channel the complex envelope
along the rotor's own trajectory. Line power = mean ``|x|²/2`` over the valid
frames minus the noise the envelope's bandwidth collects (floor density ×
bandwidth), with its SNR; the per-channel complex means (relative to a reference channel) carry
the phase needed by a propagation test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrackParams:
    fs: int = 16000
    win_s: float = 1.0
    hop_s: float = 0.5
    slew: float = 3.0
    k_max_seed: int = 40
    n_restart: int = 4
    k_max: int = 40
    f_max: float = 6000.0
    bw_hz: float = 1.0
    n_outer: int = 6
    #: Two tracks closer than this (rev/s, median over time) are one rotor.
    merge_rev_s: float = 0.1


def resample(x_ct: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    if fs_in == fs_out:
        return np.asarray(x_ct, dtype=np.float64)
    g = gcd(fs_in, fs_out)
    return resample_poly(np.asarray(x_ct, dtype=np.float64), fs_out // g, fs_in // g, axis=-1)


def _seed_channel(x: np.ndarray, fs: int) -> int:
    """The channel with the most line-like energy (spectral-flatness minimum
    over 100 Hz – 4 kHz of a 4 s segment)."""
    n = min(x.shape[1], 4 * fs)
    seg = x[:, :n] * np.hanning(n)
    pw = np.abs(np.fft.rfft(seg, axis=1)) ** 2 + 1e-30
    f = np.fft.rfftfreq(n, 1.0 / fs)
    band = (f > 100) & (f < 4000)
    p = pw[:, band]
    flat = np.exp(np.mean(np.log(p), axis=1)) / np.mean(p, axis=1)
    return int(np.argmin(flat))


def frame_grid(n_samples: int, tp: TrackParams) -> np.ndarray:
    dur = n_samples / tp.fs
    return np.arange(0.5 * tp.win_s, dur - 0.5 * tp.win_s, tp.hop_s)


def seed(x: np.ndarray, n_rotors: int, r_lo: float, r_hi: float, tp: TrackParams):
    """``(seeds (R, N), frame_times, seed channel)`` of resampled audio ``x``."""
    from tracking.comb_seed import seed_from_gram

    ft = frame_grid(x.shape[1], tp)
    ch = _seed_channel(x, tp.fs)
    seeds = seed_from_gram(
        x[ch],
        tp.fs,
        ft,
        n_rotors,
        r_lo=r_lo,
        r_hi=r_hi,
        d_grid=0.02,
        win_s=tp.win_s,
        hop_s=tp.hop_s,
        k_max=tp.k_max_seed,
        f_max=tp.f_max,
        slew=tp.slew,
        n_restart=tp.n_restart,
    )
    return seeds, ft, ch


def refine_and_profile(
    x: np.ndarray, r_init: np.ndarray, ft: np.ndarray, tp: TrackParams
) -> dict[str, Any]:
    """Coupled VK refinement of ``r_init`` on every channel of ``x`` (resampled
    to ``tp.fs``) and the per-rotor, per-order, per-channel line powers and
    relative phases read off the final envelopes."""
    from tracking.vk_tracking import VKConfig, vk_track

    cfg = VKConfig(
        fs=float(tp.fs), k_max=tp.k_max, f_max=tp.f_max, bw_hz=tp.bw_hz, n_outer=tp.n_outer
    )
    from scipy.signal import welch

    from experiments.static_rig.spectra import _floor

    x = x[:8]
    res = vk_track(x, r_init, ft, cfg)
    env = res.envelopes
    r = res.r_refined
    # one-sided noise-floor density per channel (running median of a 4 s
    # Welch): the noise a VK envelope of bandwidth B collects adds S1·B to the
    # line power estimate |x|²/2, which is subtracted below
    nper = int(min(4 * tp.fs, x.shape[1] // 2))
    fw, s1 = welch(x, fs=tp.fs, nperseg=nper, axis=-1)
    dfw = float(fw[1] - fw[0])
    floor = 10.0 ** (
        _floor(10.0 * np.log10(np.maximum(s1, 1e-30)), max(2, round(10.0 / dfw))) / 10.0
    )
    bw = env.bw_track if env.bw_track.size else np.full(env.k.shape, tp.bw_hz)
    # distinct rotors: tracks closer than merge_rev_s (median) are one rotor
    distinct: list[int] = []
    for i in np.argsort(np.median(r, axis=1)):
        if all(np.median(np.abs(r[i] - r[j])) >= tp.merge_rev_s for j in distinct):
            distinct.append(int(i))
    rotors = []
    for ri in range(r.shape[0]):
        sel = np.nonzero(env.rotor == ri)[0]
        ks, power, snr, rel = [], [], [], []
        for m in sel:
            v = env.valid[m]
            if v.sum() < 4:
                continue
            xm = env.x[:, m, v]  # (C, T_valid)
            raw = 0.5 * np.mean(np.abs(xm) ** 2, axis=1)  # real-signal power
            fc = float(env.k[m]) * float(np.mean(r[ri]))
            noise = floor[:, int(np.clip(round(fc / dfw), 0, floor.shape[1] - 1))] * float(bw[m])
            line = raw - noise
            ref = int(np.argmax(raw))
            cross = np.mean(xm * np.conj(xm[ref]), axis=1)
            coh = np.abs(cross) / np.sqrt(
                np.mean(np.abs(xm) ** 2, axis=1) * np.mean(np.abs(xm[ref]) ** 2) + 1e-300
            )
            ks.append(int(env.k[m]))
            power.append(np.where(line > 0, 10.0 * np.log10(np.maximum(line, 1e-30)), np.nan))
            snr.append(10.0 * np.log10(np.maximum(raw, 1e-30) / np.maximum(noise, 1e-30)))
            rel.append({"ref": ref, "phase": np.angle(cross).tolist(), "coherence": coh.tolist()})
        rotors.append(
            {
                "track": r[ri].tolist(),
                "mean": float(np.mean(r[ri])),
                "std": float(np.std(r[ri])),
                "range": [float(np.min(r[ri])), float(np.max(r[ri]))],
                "distinct": ri in distinct,
                "orders": ks,
                "power_db": [
                    [None if not np.isfinite(v) else float(v) for v in row] for row in power
                ],
                "snr_db": np.array(snr).tolist(),
                "phase_rel": rel,
            }
        )
    return {
        "frame_times": ft.tolist(),
        "residual_ratios": [float(v) for v in res.residual_ratios],
        "n_distinct": len(distinct),
        "rotors": rotors,
    }


def track(
    x_ct: np.ndarray, fs_in: int, n_rotors: int, r_lo: float, r_hi: float, tp: TrackParams
) -> dict[str, Any]:
    """Seed on the best channel, refine and profile on all channels."""
    x = resample(x_ct, fs_in, tp.fs)[:8]
    seeds, ft, ch = seed(x, n_rotors, r_lo, r_hi, tp)
    out = refine_and_profile(x, seeds, ft, tp)
    return {"seed_channel": ch, "seeds": seeds.tolist(), **out, "params": tp.__dict__}
