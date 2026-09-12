"""Where does the coherence information live?

The bin-level Rice likelihood was unable to recover a planted coherence
transition (planted k_half 2 and 8 both came back at ~0.7, with the full-band
NLL identical to four decimals across variants). The suspicion is dilution: the
line cells are a few hundred of ~186k band cells, so the coherent/Rayleigh
distinction is worth under 20 nats out of ~4e5.

This script separates the two possibilities without fitting anything new: it
takes the Whittle MAP, then sweeps ``k_half`` and reports the NLL over the FULL
band next to the NLL over the LINE CELLS ONLY. A profile that is flat on the
full band but has a minimum at the planted value on the line cells means the
likelihood is fine and only the weighting is wrong; a profile that is monotone
toward zero coherence on the line cells too means the bin-level Rice model is
misspecified (window leakage, sub-bin centre error) and the coherence must be
read from a band-integrated statistic instead.

Usage:
    python scripts/_coherence_profile.py --planted 8.0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from data_processing import stochastic_rotor_noise as srn
from experiments.stochastic_fit import rig as RG
from experiments.stochastic_fit import stage1_bayes as SB
from experiments.stochastic_fit.data import Clip, periodogram
from experiments.stochastic_fit.model import CombSpectrum, make_spec, measure_band_dof

SR = 16000
RATE = 70.0
SECONDS = 12.0
N_FFT = 1 << 14
K_CAP = 100


def planted_clip(k_half: float, seed: int = 5) -> Clip:
    """A clip with a KNOWN coherence transition: orders below ``k_half`` go
    through the tone bank, the rest are rendered as narrowband noise."""
    rng = np.random.default_rng(seed)
    p = srn.sample_params(
        rng, srn.StochasticRanges(), n_rotors=1, n_harmonics=K_CAP, sample_rate=SR
    )
    p = p.with_(
        coherence_k_half=k_half,
        gamma_min_bins=0.01,
        shaft_jitter_rps=0.01,
        harm_gp_std_db=0.0,
        floor_gp_std_db=0.0,
        floor_tilt_gp_std=0.0,
        umod_std_db=0.0,
        phase_diffusion_hz_per_order=0.0,
    )
    rps = np.full((1, int(SECONDS * SR)), RATE)
    audio, _ = srn.synthesize(p, rps, rng=rng, n_mics=1, line_mode="fm", n_fft=N_FFT)
    return Clip(
        f"planted_kh{k_half:g}", "planted", np.asarray(audio, np.float32), rps, SR, None, {}
    )


def line_mask(model: CombSpectrum, half_bins: float = 3.0) -> torch.Tensor:
    """``(M, N, F)`` cells within ``half_bins`` of any line centre."""
    carrier = model.carrier()  # (R, N)
    centres = model.k[None, :, None] * carrier[:, None, :]  # (R, K, N)
    d = (model.freqs[None, None, None, :] - centres[..., None]).abs()
    near = (d <= half_bins * model.df).any(dim=1).any(dim=0)  # (N, F)
    return (near[None] & model.band[None, None]).expand(model.M, -1, -1)


def rice_cells(model: CombSpectrum, power: torch.Tensor) -> torch.Tensor:
    """``(M, N, F)`` per-cell NLL under the model's own density."""
    coh, inc = model.parts()
    if coh is None:
        m = model.forward().clamp_min(1e-12)
        return power / m + torch.log(m)
    n = inc.clamp_min(1e-12)
    c = coh.clamp_min(0.0)
    z = (2.0 * torch.sqrt((power.clamp_min(0.0) * c).clamp_min(1e-30)) / n).clamp(1e-12, 1e12)
    small = torch.log(torch.special.i0e(z.clamp(max=30.0)).clamp_min(1e-30)) + z.clamp(max=30.0)
    large = z - 0.5 * torch.log(2.0 * math.pi * z)
    return torch.log(n) + (power + c) / n - torch.where(z < 30.0, small, large)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planted", type=float, default=8.0)
    ap.add_argument("--iters", type=int, nargs=3, default=(40, 40, 120))
    ap.add_argument("--out", type=Path, default=Path("results/S1/coherence_profile.json"))
    args = ap.parse_args()

    clip = planted_clip(args.planted)
    pg = periodogram(clip, n_fft=N_FFT, hop=N_FFT // 2)
    spec = make_spec(
        pg,
        n_mics=1,
        f_max=7900.0,
        k_cap=K_CAP,
        variant={**SB.BENCH_VARIANT, "fit_coherence": True},
    )
    clips = [(clip.clip_id, "planted", pg, spec)]
    rig = RG.RigParams(spec, RG.RigSpec(), pg.rps.shape[0], "cpu")
    rcs = RG._prepare(clips, rig, "cpu")
    RG._init_rig_from_clips(rig, rcs)
    t0 = time.time()
    RG._stages(
        rig,
        rcs,
        rig_free=True,
        ladder=(16, 48),
        iters=tuple(args.iters),
        lr=0.1,
        log=print,
        t0=t0,
    )
    rc = rcs[0]
    model = rc.model
    mask = line_mask(model)
    n_line = int(mask.sum().item())
    n_band = int(model.n_cells())
    print(f"\nline cells {n_line} of {n_band} band cells ({100.0 * n_line / n_band:.2f} %)")
    print(f"fitted k_half {math.exp(float(model.log_k_half[0])):.3f}")
    grid = [0.3, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0, 40.0, 100.0]
    # the band's effective dof is an ANALYSIS property, measured on line-free
    # bands of the same width
    pw = rc.power[0].cpu().numpy()
    centres = (model.k[:, None] * model.carrier()[0][None, :]).mean(dim=1).detach().cpu().numpy()
    nu = measure_band_dof(pw, model.freqs.cpu().numpy(), centres, half_bins=3.0)
    print(f"measured band dof nu = {nu:.2f} (7 bins)\n")
    rows = []
    with torch.no_grad():
        for kh in grid:
            model.log_k_half.data.fill_(math.log(kh))
            cells = rice_cells(model, rc.power)
            rows.append(
                dict(
                    k_half=kh,
                    nll_band=float(cells[..., model.band].sum().item()),
                    nll_line=float(cells[mask].sum().item()),
                    nll_bandpower=float(
                        model.band_coherence_nll(rc.power, nu=nu, half_bins=3.0).item()
                    ),
                )
            )
    b0 = min(r["nll_band"] for r in rows)
    l0 = min(r["nll_line"] for r in rows)
    p0 = min(r["nll_bandpower"] for r in rows)
    print(f"{'k_half':>8}  {'bins: full band':>16}  {'bins: line cells':>17}  {'BAND POWERS':>13}")
    for r in rows:
        print(
            f"{r['k_half']:8.1f}  {r['nll_band'] - b0:16.1f}  {r['nll_line'] - l0:17.1f}"
            f"  {r['nll_bandpower'] - p0:13.1f}"
            + ("   <- planted" if abs(r["k_half"] - args.planted) < 1e-6 else "")
        )
    best_band = min(rows, key=lambda r: r["nll_band"])["k_half"]
    best_line = min(rows, key=lambda r: r["nll_line"])["k_half"]
    best_bp = min(rows, key=lambda r: r["nll_bandpower"])["k_half"]
    print(
        f"\nargmin: bins/full {best_band:g}, bins/lines {best_line:g}, "
        f"band powers {best_bp:g}, planted {args.planted:g}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            dict(
                planted=args.planted,
                fitted_k_half=math.exp(float(model.log_k_half[0])),
                n_line_cells=n_line,
                n_band_cells=n_band,
                band_dof=nu,
                grid=rows,
                argmin_band=best_band,
                argmin_line=best_line,
                argmin_bandpower=best_bp,
            ),
            indent=2,
        )
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
