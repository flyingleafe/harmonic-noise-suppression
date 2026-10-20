"""Motor lines against aerodynamic lines in the R3 bench fits, for
``docs/explainers/noise-v4-bench-fit.qmd`` § 5.

The model puts TWO families at the motor orders ``k = 21 n`` (7 pole pairs x 3):
the aerodynamic line (profile ``p^a_k``, speed exponent ``alpha^a``, the
pedestal ``sigma2_k, gamma_k``) and the motor line (profile ``p^m_n``, exponent
``alpha^m``, the shaft alone, no pedestal). Per motor, one figure:

* top left: line power per order at 70 %, aerodynamic family vs motor family,
  with the measured line power;
* top right: power against rotor speed at the first three motor orders, each
  family and the measurement;
* bottom: zooms around the first two motor orders that are visible at 70 %,
  the data with the fit split into its aerodynamic and motor parts.

Run from the repo root::

    PYTHONPATH=src python docs/explainers/noise-v4-bench-fit/motor_lines.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build as BLD  # noqa: E402

B = BLD.B
AERO, MOTOR, DATA = "tab:orange", "tab:purple", "0.45"


def db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-30))


def family_lines(m: B.LadderModel, j: int) -> dict[str, np.ndarray]:
    """``(M, F)`` lines of each family alone at setting ``j``."""
    out = {}
    with torch.no_grad():
        pm, pa = m.params["p_motor"].clone(), m.params["p_aero"].clone()
        m.params["p_motor"].fill_(-300.0)
        out["aero"] = m.components(j)["lines"].cpu().numpy()
        m.params["p_motor"].copy_(pm)
        m.params["p_aero"].fill_(-300.0)
        out["motor"] = m.components(j)["lines"].cpu().numpy()
        m.params["p_aero"].copy_(pa)
    return out


def main() -> None:
    os.chdir(BLD.ROOT)
    summary = {}
    for motor in BLD.MOTORS:
        lad = B.build_ladder(motor)
        model, pay = BLD.load_model(lad, motor, "r3")
        calm = [m for m in range(lad.settings[0].power.shape[0]) if m not in B.WIND_MICS[motor]]
        mk = lad.motor_orders
        j70 = next(j for j, s in enumerate(lad.settings) if s.throttle == 70)
        fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))

        # per-order line power at 70 %
        with torch.no_grad():
            wa, wm, mko = model.line_weights(j70)
        wa = wa.cpu().numpy()[calm].mean(axis=0)
        wm = wm.cpu().numpy()[calm].mean(axis=0)
        meas = db((10 ** (lad.tables[j70]["level_db"][calm] / 10)).mean(axis=0))
        k = np.arange(1, wa.size + 1)
        ax[0, 0].semilogx(k, db(wa), "-", color=AERO, lw=1.2, label="aerodynamic family (fit)")
        ax[0, 0].semilogx(mko, db(wm), "s", color=MOTOR, ms=8, label="motor family (fit)")
        ax[0, 0].semilogx(
            np.arange(1, meas.size + 1), meas, ".", color=DATA, ms=4, label="measured line power"
        )
        for kk in mko:
            ax[0, 0].axvline(kk, color=MOTOR, lw=0.6, alpha=0.4)
        ax[0, 0].set_title(f"{motor} 70 %: line power per order (calm-mic mean)")
        ax[0, 0].set_xlabel("order k")
        ax[0, 0].set_ylabel("line power (dB)")
        ax[0, 0].legend(fontsize=9)

        # power against speed at the first three motor orders
        car = np.array([s.carrier for s in lad.settings])
        for i, kk in enumerate(mk[:3]):
            a_lv, m_lv, d_lv = [], [], []
            for j, s in enumerate(lad.settings):
                with torch.no_grad():
                    wa_j, wm_j, mk_j = model.line_weights(j)
                a_lv.append(
                    db(wa_j.cpu().numpy()[calm, kk - 1].mean()) if kk <= s.k_max else np.nan
                )
                m_lv.append(db(wm_j.cpu().numpy()[calm, i].mean()) if i < mk_j.size else np.nan)
                d_lv.append(
                    db((10 ** (lad.tables[j]["level_db"][calm, kk - 1] / 10)).mean())
                    if kk <= lad.tables[j]["level_db"].shape[1]
                    else np.nan
                )
            ls = ("-", "--", ":")[i]
            ax[0, 1].plot(
                car, a_lv, ls, color=AERO, marker="o", ms=4, label=f"k = {kk}: aerodynamic"
            )
            ax[0, 1].plot(car, m_lv, ls, color=MOTOR, marker="s", ms=4, label=f"k = {kk}: motor")
            ax[0, 1].plot(car, d_lv, ls, color=DATA, marker=".", ms=6, label=f"k = {kk}: measured")
        ax[0, 1].set_title(
            f"against speed: α^a = {pay['shared']['alpha_aero']:.2f}, α^m = {pay['shared']['alpha_motor']:.2f}"
        )
        ax[0, 1].set_xlabel("rotor speed (rev/s)")
        ax[0, 1].set_ylabel("line power (dB)")
        ax[0, 1].legend(fontsize=7, ncol=3)

        # zooms at the motor orders, 70 %
        s = lad.settings[j70]
        f = s.freqs
        fam = family_lines(model, j70)
        pr = BLD.parts(model, j70)
        rest = pr["floor"] + pr["wind"] + pr["ambient"]
        kern = np.ones(5) / 5.0
        dsm = np.convolve(s.power[calm].mean(axis=0), kern, mode="same")
        for a, kk in zip(ax[1], mk[:2]):
            sel = np.abs(f - kk * s.carrier) <= 1.5 * s.carrier
            a.plot(f[sel], db(dsm[sel]), color=DATA, lw=0.7, label="data (5-bin mean)")
            a.plot(
                f[sel],
                db((fam["aero"] + fam["motor"] + rest)[calm].mean(axis=0)[sel]),
                "k",
                lw=1.4,
                label="fit total",
            )
            a.plot(
                f[sel],
                db(fam["aero"][calm].mean(axis=0)[sel]),
                color=AERO,
                lw=1.2,
                label="aerodynamic lines (core + pedestal)",
            )
            a.plot(
                f[sel],
                db(fam["motor"][calm].mean(axis=0)[sel]),
                color=MOTOR,
                lw=1.2,
                label="motor lines (shaft only)",
            )
            top = db(dsm[sel]).max()
            a.set_ylim(top - 50, top + 5)
            a.set_xlim(f[sel][0], f[sel][-1])
            a.axvline(kk * s.carrier, color=MOTOR, lw=0.6, alpha=0.4)
            a.set_title(
                f"orders {kk - 1}–{kk + 1} at 70 % (motor order k = {kk} at {kk * s.carrier:.0f} Hz)"
            )
            a.set_xlabel("frequency (Hz)")
            a.set_ylabel("power (dB)")
        ax[1, 0].legend(fontsize=8, loc="lower left")
        for a in ax.flat:
            a.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(HERE / f"motor_lines_{motor}.png", dpi=120)
        plt.close(fig)

        s2 = np.asarray(pay["shared"]["pedestal"]["sigma2"])
        gam = np.asarray(pay["shared"]["pedestal"]["gamma_hz"])
        pa_ = np.asarray(pay["shared"]["profile_aero_db"])
        pm_ = np.asarray(pay["shared"]["profile_motor_db"])
        summary[motor] = dict(
            alpha_aero=pay["shared"]["alpha_aero"],
            alpha_motor=pay["shared"]["alpha_motor"],
            orders=[
                dict(
                    k=int(kk),
                    p_aero_db=float(pa_[kk - 1]),
                    p_motor_db=float(pm_[i]),
                    motor_minus_aero_db=float(pm_[i] - pa_[kk - 1]),
                    aero_sigma2=float(s2[kk - 1]),
                    aero_core_share=float(math.exp(-s2[kk - 1])),
                    aero_gamma_hz=float(gam[kk - 1]),
                )
                for i, kk in enumerate(mk[:6])
            ],
        )
        print(motor, json.dumps(summary[motor], indent=1))
    (HERE / "motor_lines.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
