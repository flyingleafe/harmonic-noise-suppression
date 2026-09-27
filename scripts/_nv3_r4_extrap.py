"""Throwaway: how a round-4 cruise fit extrapolates below cruise under its own
speed laws against the v2 short-span pin, next to Michael's standby fit.

    PYTHONPATH=src python scripts/_nv3_r4_extrap.py \\
        --dregon results/noise_v3/fits_r4b/dregon_room2_floor__flight_v3.json \\
        --cruise results/noise_v3/fits_r4c/michaels_fly125_cruise__flight_v3.json \\
        --standby results/noise_v3/fits_r4c/michaels_fly125_standby__flight_v3.json

Prints, per fit and speed-law policy, the mic-mean expected periodogram's band
levels (dB, the LTAS bands) with every rotor at 35 / 50 / 65 / 80 rev/s, and the
line-over-floor at k = 1, 2. Expectations only (light).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from experiments.noise_model import render as RD
from experiments.noise_model import rig_sampler as RS

BANDS = ((100, 300), (300, 700), (700, 1500), (1500, 3000), (3000, 5000), (5000, 7000))
SPEEDS = (35.0, 50.0, 65.0, 80.0)
SR, N_FFT, HOP = 16000, 2048, 512


def bands_db(fit: dict, rps: float) -> tuple[np.ndarray, float, float]:
    r = np.full((4, N_FFT * 4), rps)
    m = RD.expected_periodogram(fit, r, n_fft=N_FFT, hop=HOP, sr=SR, n_mics=8).mean(axis=(0, 1))
    off = copy.deepcopy(fit)
    off["params"]["profile"]["profile_db"] = (
        np.asarray(fit["params"]["profile"]["profile_db"]) * 0.0 - 300.0
    ).tolist()
    fl = RD.expected_periodogram(off, r, n_fft=N_FFT, hop=HOP, sr=SR, n_mics=8).mean(axis=(0, 1))
    f = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    out = np.array([10 * np.log10(m[(f >= lo) & (f < hi)].mean()) for lo, hi in BANDS])
    k1 = int(round(rps / (SR / N_FFT)))
    lof = [10 * np.log10(m[k * k1 - 1 : k * k1 + 2].max() / fl[k * k1]) for k in (1, 2)]
    return out, float(lof[0]), float(lof[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dregon", required=True)
    ap.add_argument("--cruise", required=True)
    ap.add_argument("--standby", required=True)
    args = ap.parse_args()
    fits = {
        "dregon": json.loads(Path(args.dregon).read_text()),
        "cruise": json.loads(Path(args.cruise).read_text()),
        "standby": json.loads(Path(args.standby).read_text()),
    }
    print("bands:", BANDS)
    for name, fit in fits.items():
        p = fit["params"]
        laws = {
            "fitted": (
                p["profile"]["amp_exp"],
                p["floor"]["floor_exp"],
                p["floor"]["floor_static_rel"],
            ),
            "span_pin": (
                RS.SPAN_PIN["amp_exp"],
                RS.SPAN_PIN["floor_exp"],
                RS.SPAN_PIN["floor_static_rel"],
            ),
        }
        for law, (a, e, s) in laws.items():
            g = copy.deepcopy(fit)
            g["params"]["profile"]["amp_exp"] = a
            g["params"]["floor"]["floor_exp"] = e
            g["params"]["floor"]["floor_static_rel"] = s
            print(f"\n{name} [{law}: amp {a:.2f} floor {e:.3f} static {s:.2e}]")
            for rps in SPEEDS:
                b, k1, k2 = bands_db(g, rps)
                print(
                    f"  {rps:4.0f} rev/s  "
                    + " ".join(f"{v:6.1f}" for v in b)
                    + f"   k1 {k1:+5.1f} k2 {k2:+5.1f}"
                )


if __name__ == "__main__":
    main()
