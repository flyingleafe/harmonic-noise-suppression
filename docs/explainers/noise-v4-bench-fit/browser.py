"""The interactive parameter view of ``docs/explainers/noise-v4-bench-fit.qmd``:
the same design as ``noise_lab.rig_view_plotly`` (expected periodogram, stems at
``k f_r`` from the floor to the line level, red caps of +/- a half width,
dropdown + speed slider), for the v4 BENCH fits — one dropdown entry per
(motor, fit), the slider over the five recorded throttle settings, the data on
the same axes.

Everything is binned to 3 Hz (mean power per bin, calm-mic mean) so the page
stays small; the stems read the binned model at the bin of ``k f_r``. The caps
are +/- the pedestal half-width ``gamma_k``.

Run from the repo root::

    PYTHONPATH=src python docs/explainers/noise-v4-bench-fit/browser.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build as BLD  # noqa: E402

B = BLD.B
BIN_HZ = 3.0


def binned(f: np.ndarray, x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.digitize(f, edges) - 1
    ok = (idx >= 0) & (idx < edges.size - 1)
    tot = np.bincount(idx[ok], weights=x[ok], minlength=edges.size - 1)
    n = np.bincount(idx[ok], minlength=edges.size - 1)
    return tot / np.maximum(n, 1)


def main() -> None:
    os.chdir(BLD.ROOT)
    edges = np.arange(B.BAND_HZ[0], B.BAND_HZ[1] + BIN_HZ, BIN_HZ)
    fc = 0.5 * (edges[:-1] + edges[1:])
    views = []
    for motor in BLD.MOTORS:
        lad = B.build_ladder(motor)
        calm = [m for m in range(lad.settings[0].power.shape[0]) if m not in B.WIND_MICS[motor]]
        for arm in BLD.ARMS:
            model, pay = BLD.load_model(lad, motor, arm)
            gam = np.asarray(pay["shared"]["pedestal"]["gamma_hz"], dtype=float)
            s2 = np.asarray(pay["shared"]["pedestal"]["sigma2"], dtype=float)
            steps = []
            for j, s in enumerate(lad.settings):
                pr = BLD.parts(model, j)
                curves = {
                    name: 10
                    * np.log10(
                        np.maximum(binned(s.freqs, pr[name][calm].mean(axis=0), edges), 1e-30)
                    )
                    for name in ("total", "core", "pedestal", "floor", "ambient")
                }
                curves["data"] = 10 * np.log10(binned(s.freqs, s.power[calm].mean(axis=0), edges))
                k = np.arange(1, s.k_max + 1)
                fk = k * s.carrier
                b = np.clip(np.digitize(fk, edges) - 1, 0, fc.size - 1)
                st = pay["settings"][j]
                steps.append(
                    dict(
                        throttle=s.throttle,
                        carrier=s.carrier,
                        sigma_nu=st["sigma_nu_rad_s"],
                        lam=st["lam_per_s"],
                        curves={n: np.round(v, 2).tolist() for n, v in curves.items()},
                        fk=np.round(fk, 2).tolist(),
                        line_db=np.round(curves["total"][b], 2).tolist(),
                        floor_db=np.round(curves["floor"][b], 2).tolist(),
                        gamma=np.round(gam[: s.k_max], 3).tolist(),
                        sigma2=np.round(s2[: s.k_max], 3).tolist(),
                    )
                )
            views.append(
                dict(
                    label=f"{motor} · {arm.upper()}",
                    alpha=pay["shared"]["alpha_aero"],
                    objective=pay["objective"]["total_nats"],
                    steps=steps,
                )
            )
            print(motor, arm, "done")
    (HERE / "browser.json").write_text(
        json.dumps(dict(freqs=np.round(fc, 2).tolist(), views=views))
    )


if __name__ == "__main__":
    main()
