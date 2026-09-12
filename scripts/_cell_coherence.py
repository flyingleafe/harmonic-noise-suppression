"""Cell-level vs band-level line statistics: what decoheres, and what a band hides.

A tone on a shaft whose phase is Brownian with per-sample increment ``sigma``
accumulates ``k * sigma * sqrt(n_fft)`` radians of phase across one analysis
window, so order ``k`` decoheres at ``k^2``: its centre CELL goes from
deterministic to Rayleigh (power CV 0 -> 1) and its mean level collapses. The
BAND power around the same line is conserved as long as the broadened line fits
inside the band, so a band-integrated CV reads ~0 in exactly that regime. The
two statistics therefore answer different questions, and comparing one against
the other is how a phase-diffusion mechanism gets mistaken for amplitude noise.

This script measures both, per order, on the same clips:
  * the real bench cell (native 44.1 kHz, decimated to 16 kHz, channel 7)
  * ``fm`` renders with a shared OU shaft jitter (the phase-diffusion mechanism)
  * a ``stochastic`` render (every line Rayleigh by construction)

Usage:
    python scripts/_cell_coherence.py --motor 1 --speed 80
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_processing import stochastic_rotor_noise as srn
from experiments.stochastic_fit import stage1_bayes as SB

N_FFT = 1 << 14
HOP = N_FFT // 2
BAND_HZ = 6.0  # default; --band overrides (a band narrower than the
# line CENSORS the equivalent width, which is how a k-linear pedestal can look
# saturated)
ORDERS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def stft_power(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """``(N, F)`` per-frame periodogram, Hann window."""
    win = np.hanning(n_fft)
    starts = range(0, max(len(x) - n_fft, 0) + 1, hop)
    frames = np.stack([x[i : i + n_fft] * win for i in starts])
    return np.abs(np.fft.rfft(frames, axis=-1)) ** 2


def line_stats(
    x: np.ndarray, sr: int, rate: float, orders=ORDERS, band_hz: float = BAND_HZ
) -> dict[int, dict[str, float]]:
    """Per order: CV of the centre cell, of the +-BAND_HZ band, and the width.

    The width is what ties the two CVs to a mechanism. Phase diffusion of half
    width ``gamma`` spreads a line over ``~2 gamma``: while that fits inside the
    band, band power is conserved (band CV -> 0) and only the CELL decoheres;
    once it exceeds the band, the band sees a slice of a Rayleigh field and its
    CV rises to ``1 / sqrt(nu)``. So a real line with a high band CV is either
    broad (diffused past the band) or amplitude-random, and the measured width
    says which.

    ``w_eq_hz`` is the equivalent width of the frame-averaged excess over a
    local floor; ``frac_1bin`` is the share of that excess in the centre bin.
    """
    P = stft_power(x)
    f = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    mean_p = P.mean(axis=0)
    df = float(f[1] - f[0])
    out: dict[int, dict[str, float]] = {}
    for k in orders:
        fc = k * rate
        if fc > f[-1] - band_hz:
            continue
        cell = P[:, int(np.argmin(np.abs(f - fc)))]
        in_band = np.abs(f - fc) <= band_hz
        band = P[:, in_band].sum(axis=1)
        annulus = (np.abs(f - fc) > band_hz) & (np.abs(f - fc) <= 1.6 * band_hz)
        floor = float(np.median(mean_p[annulus])) if annulus.any() else 0.0
        excess = np.clip(mean_p[in_band] - floor, 0.0, None)
        peak = float(excess.max()) if excess.size else 0.0
        centre = float(excess[int(np.argmin(np.abs(f[in_band] - fc)))]) if peak > 0 else 0.0
        out[k] = dict(
            cv_cell=float(cell.std() / max(cell.mean(), 1e-30)),
            cv_band=float(band.std() / max(band.mean(), 1e-30)),
            w_eq_hz=float(excess.sum() * df / peak) if peak > 0 else float("nan"),
            frac_1bin=float(centre / max(excess.sum(), 1e-30)),
            level_db=float(10.0 * np.log10(max(band.mean(), 1e-30))),
        )
    return out


def render(
    *,
    sr: int,
    rate: float,
    seconds: float,
    line_mode: str,
    shaft_jitter: float,
    seed: int = 11,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = srn.sample_params(rng, srn.StochasticRanges(), n_rotors=1, n_harmonics=120, sample_rate=sr)
    p = p.with_(
        shaft_jitter_rps=shaft_jitter,
        shaft_jitter_tau_s=0.05,
        phase_diffusion_hz_per_order=0.0,
        coherence_k_half=0.0,
        gamma_min_bins=0.01,
        harm_gp_std_db=0.0,
        floor_gp_std_db=0.0,
        floor_tilt_gp_std=0.0,
        umod_std_db=0.0,
        floor_mean_db=p.floor_mean_db - 40.0,
    )
    rps = np.full((1, int(seconds * sr)), rate)
    audio, _ = srn.synthesize(p, rps, rng=rng, n_mics=1, line_mode=line_mode, n_fft=N_FFT)
    return np.asarray(audio, float).ravel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motor", type=int, default=1)
    ap.add_argument("--speed", type=int, default=80)
    ap.add_argument("--band", type=float, default=BAND_HZ)
    ap.add_argument("--out", type=Path, default=Path("results/S1/cell_coherence.json"))
    args = ap.parse_args()

    clip = SB.bench_clip(args.motor, args.speed)
    x = np.asarray(clip.audio, float)
    x = x[0] if x.ndim > 1 else x
    rate = float(np.asarray(clip.rps).mean())
    seconds = len(x) / clip.sr
    print(f"real bench Motor{args.motor}_{args.speed}: {seconds:.1f} s at {rate:.2f} rev/s\n")

    cases = {"real": line_stats(x, clip.sr, rate, band_hz=args.band)}
    for sj in (0.0, 0.003, 0.01, 0.03):
        cases[f"fm sj={sj:g}"] = line_stats(
            render(
                sr=clip.sr,
                rate=rate,
                seconds=seconds,
                line_mode="fm",
                shaft_jitter=sj,
            ),
            clip.sr,
            rate,
            band_hz=args.band,
        )
    cases["stochastic"] = line_stats(
        render(
            sr=clip.sr,
            rate=rate,
            seconds=seconds,
            line_mode="stochastic",
            shaft_jitter=0.0,
        ),
        clip.sr,
        rate,
    )

    ks = sorted(set().union(*[set(v) for v in cases.values()]))
    for stat, title in (
        ("cv_cell", "CENTRE CELL power CV (0 = tone, 1 = Rayleigh)"),
        ("cv_band", f"+-{args.band:g} Hz BAND power CV (0 = conserved, 1/sqrt(nu) = Rayleigh)"),
        ("w_eq_hz", "EQUIVALENT WIDTH of the line (Hz; one bin = %.2f Hz)" % (16000 / N_FFT)),
        ("frac_1bin", "share of the line's excess in the CENTRE BIN"),
    ):
        print(title)
        print("  " + "case".ljust(14) + "".join(f"k{k:<5d}" for k in ks))
        for name, st in cases.items():
            row = "".join(f"{st[k][stat]:<6.2f}" if k in st else "  --  " for k in ks)
            print("  " + name.ljust(14) + row)
        print()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({k: {str(a): b for a, b in v.items()} for k, v in cases.items()}, indent=2)
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
