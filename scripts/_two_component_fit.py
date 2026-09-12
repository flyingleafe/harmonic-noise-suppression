"""Does the two-component line fit the bench better, and does it reproduce the
measured width curve?

The bench's line shape is a mixture of a window-limited needle and a pedestal:
the measured equivalent width rises from 1.50 Hz at k = 1 to ~5.3 Hz at k >= 24
while the centre-bin share falls 0.65 -> 0.17, and both statistics imply the
same coherent fraction to within 0.005. That mixture is in the MEAN spectrum, so
this is a plain Whittle comparison of two nested models -- one component versus
two -- with no bespoke statistic anywhere.

Reported per variant: the Whittle NLL per cell, the excess over the leave-one-out
reference, the fitted ``k_half``, and the model's own equivalent width per order
against the measured one.

Usage:
    python scripts/_two_component_fit.py --motor 1 --speed 80
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.stochastic_fit import rig as RG
from experiments.stochastic_fit import stage1_bayes as SB
from experiments.stochastic_fit.data import periodogram
from experiments.stochastic_fit.model import CombSpectrum, make_spec

ORDERS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
BAND_HZ = 30.0  # must exceed the widest line, or the equivalent width is censored


def model_widths(model: CombSpectrum, orders=ORDERS) -> dict[int, float]:
    """Equivalent width of each of the model's own lines, measured the same way
    as on the clip: band excess over a local floor, divided by the peak."""
    with torch.no_grad():
        spectrum = model.forward()[0].mean(dim=0).cpu().numpy()  # (F,)
        floor = model.floor()[0].mean(dim=0).cpu().numpy()
        f = model.freqs.cpu().numpy()
        rate = float(model.carrier()[0].mean().item())
    df = float(f[1] - f[0])
    out: dict[int, float] = {}
    for k in orders:
        fc = k * rate
        sel = np.abs(f - fc) <= BAND_HZ
        if sel.sum() < 3:
            continue
        excess = np.clip(spectrum[sel] - floor[sel], 0.0, None)
        peak = float(excess.max())
        out[k] = float(excess.sum() * df / peak) if peak > 0 else float("nan")
    return out


def fit_one(clip, pg, variant: dict, iters: tuple[int, int, int]) -> dict:
    spec = make_spec(pg, n_mics=1, f_max=7900.0, k_cap=200, variant={**SB.BENCH_VARIANT, **variant})
    clips = [(clip.clip_id, clip.group, pg, spec)]
    rig = RG.RigParams(spec, RG.RigSpec(), pg.rps.shape[0], "cpu")
    rcs = RG._prepare(clips, rig, "cpu")
    RG._init_rig_from_clips(rig, rcs)
    RG._stages(
        rig,
        rcs,
        rig_free=True,
        ladder=(16, 48),
        iters=iters,
        lr=0.1,
        log=print,
        t0=time.time(),
    )
    rc = rcs[0]
    scores = RG._scores(rc)
    kh = math.exp(float(rc.model.log_k_half[0])) if rc.model.spec.fit_coherence else 0.0
    with torch.no_grad():
        w = rc.model.coherent_fraction()
        w_k = None if w is None else w[0, :, 0].cpu().numpy()
    return dict(scores=scores, k_half=kh, widths=model_widths(rc.model), w_k=w_k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motor", type=int, default=1)
    ap.add_argument("--speed", type=int, default=80)
    ap.add_argument("--iters", type=int, nargs=3, default=(40, 40, 150))
    ap.add_argument("--out", type=Path, default=Path("results/S1/two_component.json"))
    args = ap.parse_args()

    clip = SB.bench_clip(args.motor, args.speed)
    pg = periodogram(clip, n_fft=SB.N_FFT, hop=SB.HOP)
    measured = json.loads(Path("results/S1/cell_coherence_b30.json").read_text())["real"]

    res = {}
    for tag, variant in (
        ("one component", {}),
        ("two components", {"fit_coherence": True}),
        ("needle = |W|^2", {"fit_coherence": True, "needle_window_shape": True}),
    ):
        print(f"\n=== {tag} ===")
        res[tag] = fit_one(clip, pg, variant, tuple(args.iters))

    print(f"\n{'':16}" + "".join(f"{'nll/cell':>12}{'excess':>9}{'k_half':>8}"))
    for tag, r in res.items():
        s = r["scores"]
        print(f"{tag:16}{s['nll_fit']:12.4f}{s['excess_over_loo']:9.3f}{r['k_half']:8.2f}")
    print(f"\nequivalent width per order (Hz), one bin = {16000 / SB.N_FFT:.2f} Hz")
    ks = [k for k in ORDERS if str(k) in measured]
    print("  " + "case".ljust(16) + "".join(f"k{k:<5d}" for k in ks))
    print("  " + "measured".ljust(16) + "".join(f"{measured[str(k)]['w_eq_hz']:<6.2f}" for k in ks))
    for tag, r in res.items():
        print(
            "  " + tag.ljust(16) + "".join(f"{r['widths'].get(k, float('nan')):<6.2f}" for k in ks)
        )
    key0 = "needle = |W|^2"
    if res[key0]["w_k"] is not None:
        wk = res[key0]["w_k"]
        print("\n  fitted w_k".ljust(18) + "".join(f"{wk[k - 1]:<6.2f}" for k in ks))
        print(
            "  implied (width)".ljust(18)
            + "".join(
                f"{(1.0 / measured[str(k)]['w_eq_hz'] - 1 / 6.0) / (1 / 1.5 - 1 / 6.0):<6.2f}"
                for k in ks
            )
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                tag: dict(
                    scores=r["scores"],
                    k_half=r["k_half"],
                    widths={str(k): v for k, v in r["widths"].items()},
                    w_k=None if r["w_k"] is None else [float(v) for v in r["w_k"]],
                )
                for tag, r in res.items()
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
