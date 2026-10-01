"""Figures for the explainer: the measured pedestal of DREGON Motor1_70 vs
order (table from the 2026-10-01 single-rotor analysis, /tmp/dregon_pedestal.py)
and a sketch of the AM + FM line shape."""

from pathlib import Path

import numpy as np

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(exist_ok=True)

# Motor1_70, T = 29 s: pedestal level (dB re core peak) at |offset| Hz, mic-median.
OFFS = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
PED = {
    1: [-26.3, -30.3, -35.8, -39.6, -41.4, -42.9, -44.2, -42.7],
    2: [-23.8, -29.2, -37.4, -40.3, -47.3, -54.9, -56.8, -57.4],
    4: [-17.4, -18.5, -28.4, -33.1, -38.7, -43.4, -47.9, -52.2],
    8: [-9.0, -9.2, -14.7, -21.3, -25.4, -31.8, -36.1, -39.3],
    16: [-9.0, -7.9, -12.9, -15.3, -19.6, -25.2, -30.2, -35.2],
    32: [-7.3, -6.6, -6.6, -7.7, -8.5, -10.5, -14.8, -20.1],
}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, v in PED.items():
        ax[0].plot(OFFS, v, "o-", label=f"k = {k}")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("offset from the line centre (Hz)")
    ax[0].set_ylabel("level re line peak (dB)")
    ax[0].set_title("Measured pedestal, DREGON Motor1_70 (T = 29 s)")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    # sketch: carrier Lorentzian (FM) + AM pedestal, for three orders
    f = np.linspace(-6, 6, 4001)
    D, lam_m, sig_m2 = 1e-4, 2.5, 0.03  # rev^2/s, 1/s, modulation power
    for k, c in ((2, "C0"), (12, "C1"), (30, "C2")):
        g = np.pi * k * k * D
        g = max(g, 0.034)  # resolution of a 29 s record
        lor = (g / np.pi) / (f**2 + g**2)
        gm = lam_m / (2 * np.pi)
        ped = sig_m2 * (gm / np.pi) / (f**2 + gm**2)
        # pedestal smeared by the carrier Lorentzian: Lorentzians add widths
        gp = g + gm
        ped = sig_m2 * (gp / np.pi) / (f**2 + gp**2)
        tot = lor + ped
        ax[1].plot(
            f,
            10 * np.log10(tot / tot.max()),
            color=c,
            label=f"k = {k}: γ_k = {np.pi * k * k * D:.3f} Hz",
        )
    ax[1].set_ylim(-60, 2)
    ax[1].set_xlabel("offset from the line centre (Hz)")
    ax[1].set_ylabel("dB re peak")
    ax[1].set_title(
        "Model: Lorentzian carrier + AM pedestal\n(D = 1e-4 rev²/s, λ_m = 2.5 s⁻¹, m_rms² = 3 %)"
    )
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "pedestal.png", dpi=130)


if __name__ == "__main__":
    main()
