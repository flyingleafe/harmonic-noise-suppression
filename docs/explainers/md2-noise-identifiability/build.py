"""Small, deterministic identifiability checks using the flight fit's own kernels.

Run from the repository root:
    python docs/explainers/md2-noise-identifiability/build.py

No audio loading, model fitting, or changes to accepted fit payloads. The isolated
line is a controlled example, not an estimate of MD2's real line widths.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.noise_model import flight_v4 as flight
from experiments.noise_model import spectrum, v4_priors

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FIT_DIR = ROOT / "results/noise_v4/flight/restarts"
FIT_NAMES = {
    "standby": "md2_fly125_standby_fpb__flight_v4__s0__r6c1.json",
    "cruise": "md2_fly125_cruise_fpb__flight_v4__s0__r6c3.json",
}


def main() -> None:
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    grid = spectrum.flight_grid(sr=16000, n_fft=2048, hop=512, sr_work=32000, apply_transfer=False)
    full_grid = spectrum.flight_grid(
        sr=32000, n_fft=4096, hop=1024, sr_work=32000, apply_transfer=False
    )
    order = torch.tensor([40.0], dtype=torch.float64)
    rates = torch.full((1, 1, grid.n_fft_work), 80.0, dtype=torch.float64)
    phase, speed = flight._phase_speed(grid, rates)
    atom, _ = flight._unit_autocorr(
        phase, grid.window_work * speed.sqrt(), order, grid.n_fft_work, None
    )

    def line(
        s2a: float,
        ga: float,
        s2p: float,
        gp: float,
        sigma: float = 0.0,
        lam: float = v4_priors.FLIGHT_LAM,
        full_band: bool = False,
    ) -> np.ndarray:
        parameters = [torch.tensor([x], dtype=torch.float64) for x in (s2a, ga, s2p, gp)]
        pedestal = flight._pedestal_lag_t(grid, *parameters)
        shaft = flight._shaft_frames(
            order, torch.tensor([[sigma**2]], dtype=torch.float64), flight._v_unit(grid, lam)
        )
        output_grid = full_grid if full_band else grid
        result = flight._lag_to_bins(output_grid, 2.0 * atom * pedestal[None, :, None, :] * shaft)
        result = result.detach().numpy().reshape(-1)
        assert result.min() >= -1e-9, result.min()
        return np.maximum(result, 0.0)

    def phase_line(variance: float, width: float) -> np.ndarray:
        gamma = width * (-np.expm1(-variance)) / variance
        return line(0.12, 0.14, variance, gamma)

    # The entire windowed operator is invariant to exchanging the two OU factors.
    swapped_a = line(0.12, 0.14, 1.0, 5.0)
    swapped_b = line(1.0, 5.0, 0.12, 0.14)
    swap_error = float(np.max(np.abs(swapped_a - swapped_b)) / swapped_a.max())
    assert swap_error < 1e-12
    equal_rate_a = line(0.2, 3.0, 0.8, 3.0)
    equal_rate_b = line(0.7, 3.0, 0.3, 3.0)
    equal_rate_error = float(np.max(np.abs(equal_rate_a - equal_rate_b)) / equal_rate_a.max())
    assert equal_rate_error < 1e-12

    widths = (1.0, 8.0, 24.0)
    spectra = [phase_line(1.0, width) for width in widths]
    sums = np.array([s.sum() for s in spectra])
    # The process has fixed mean power (rho(0) = 1), but this operator retains
    # only the output's 0--8 kHz band; broadening can move some power outside it.
    full_powers = []
    for width in widths:
        values = line(0.12, 0.14, 1.0, width * (-np.expm1(-1.0)), full_band=True)
        full_powers.append(values[0] + values[-1] + 2 * values[1:-1].sum())
    full_powers = np.asarray(full_powers)
    assert np.allclose(full_powers, full_powers[0], rtol=1e-10), full_powers
    frequency_offset = np.asarray(grid.freqs_hz) - 3200.0
    region = np.abs(frequency_offset) <= 90.0
    peak = max(s.max() for s in spectra)
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 170}
    )
    fig, ax = plt.subplots(figsize=(8.2, 3.6), constrained_layout=True)
    for width, values in zip(widths, spectra, strict=True):
        ax.plot(
            frequency_offset[region],
            10 * np.log10(np.maximum(values[region] / peak, 1e-12)),
            marker="o",
            ms=3.5,
            lw=1.7,
            label=f"Sideband width W = {width:g} Hz",
        )
    ax.set(
        xlabel="Frequency offset from the 3.2 kHz carrier (Hz)",
        ylabel="Expected bin power\n(dB, common reference)",
        ylim=(-45, 1),
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)
    fig.savefig(OUT / "windowed-widths.png")
    plt.close(fig)

    # An expected Whittle contrast, not a fit to real MD2 data. Gain and floor are
    # known here, so this deliberately isolates information in the line shape.
    selected = np.abs(frequency_offset) <= 65.0
    truth = phase_line(1.0, 8.0)[selected]
    known_floor = 0.01 * truth.max()
    truth = truth + known_floor
    variances = np.geomspace(0.1, 2.8, 28)
    trial_widths = np.geomspace(0.25, 24.0, 32)
    contrast = np.empty((variances.size, trial_widths.size))
    for i, variance in enumerate(variances):
        for j, width in enumerate(trial_widths):
            candidate = phase_line(float(variance), float(width))[selected] + known_floor
            ratio = truth / candidate
            contrast[i, j] = np.sum(ratio - np.log(ratio) - 1.0)
    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    image = ax.pcolormesh(
        trial_widths, variances, np.log10(1.0 + contrast), shading="auto", cmap="viridis"
    )
    ax.plot(8.0, 1.0, marker="x", color="white", ms=10, mew=2, label="Known generating values")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Phase-sideband width W (Hz)",
        ylabel="Phase variance (rad²)",
    )
    ax.legend(loc="lower left", fontsize=9, facecolor="0.2", labelcolor="white", framealpha=0.9)
    fig.colorbar(image, ax=ax, label="log₁₀(1 + expected Whittle contrast)")
    fig.savefig(OUT / "shape-information.png")
    plt.close(fig)

    def log_mean(log_parameters: np.ndarray) -> np.ndarray:
        variance, width = np.exp(log_parameters)
        return np.log(phase_line(float(variance), float(width))[selected] + known_floor)

    centre = np.log([1.0, 8.0])
    epsilon = 1e-4
    jacobian = np.stack(
        [
            (log_mean(centre + epsilon * np.eye(2)[j]) - log_mean(centre - epsilon * np.eye(2)[j]))
            / (2 * epsilon)
            for j in range(2)
        ],
        axis=1,
    )
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    assert singular_values[-1] > 1e-3
    amp_small = line(0.0, 0.14, 1.0, 8.0 * (-np.expm1(-1.0)))
    amp_held = phase_line(1.0, 8.0)
    amp_shape_difference = float(np.max(np.abs(amp_small - amp_held)) / amp_held.max())
    shaft_a = line(0.12, 0.14, 1.0, 5.0, sigma=3.0, lam=50.0)
    shaft_b = line(0.12, 0.14, 1.0, 5.0, sigma=3.0, lam=200.0)
    shaft_difference = float(np.max(np.abs(shaft_a - shaft_b)) / shaft_a.max())
    assert shaft_difference > 0.0

    fit_records = {}
    for regime, name in FIT_NAMES.items():
        path = FIT_DIR / name
        payload_bytes = path.read_bytes()
        payload = json.loads(payload_bytes)
        pedestal = payload["params"]["pedestal"]
        width = np.asarray(pedestal["width_phase_hz"])
        fit_records[regime] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "phase_fitted": not pedestal["held"]["phase"],
            "amplitude_held": pedestal["held"]["amplitude"],
            "orders": int(width.size),
            "phase_width_hz_min_median_max": [
                float(width.min()),
                float(np.median(width)),
                float(width.max()),
            ],
            "phase_width_above_bin_count": int(np.sum(width > 16000 / 2048)),
            "phase_width_near_cap_count": int(np.sum(width >= 0.99 * v4_priors.W_MAX_HZ)),
            "gamma_amp_hz_min_max": [
                float(np.min(pedestal["gamma_amp_hz"])),
                float(np.max(pedestal["gamma_amp_hz"])),
            ],
            "sigma2_amp_min_max": [
                float(np.min(pedestal["sigma2_amp"])),
                float(np.max(pedestal["sigma2_amp"])),
            ],
            "phase_width_source": pedestal["source"]["phase"],
        }

    source_hashes = {}
    for module in (flight, spectrum, v4_priors):
        source_file = module.__file__
        assert source_file is not None
        source_path = Path(source_file)
        source_hashes[str(source_path.relative_to(ROOT))] = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()

    report = {
        "scope": "Deterministic isolated-line operator checks; no real-data fit or confidence interval",
        "operator": "flight_v4._unit_autocorr + _pedestal_lag_t + _shaft_frames + _lag_to_bins",
        "operator_source_sha256": source_hashes,
        "grid": {
            "sr": 16000,
            "n_fft": 2048,
            "sr_work": 32000,
            "window_s": 0.128,
            "bin_hz": 7.8125,
            "hann_enbw_hz": 11.71875,
        },
        "controlled_line": {
            "rotor_hz": 80,
            "order": 40,
            "carrier_hz": 3200,
            "sigma2_amp": 0.12,
            "gamma_amp_hz": 0.14,
            "phase_variance": 1.0,
            "phase_widths_hz": list(widths),
        },
        "checks": {
            "am_pm_swap_relative_error": swap_error,
            "equal_rate_variance_split_relative_error": equal_rate_error,
            "captured_band_power_relative_spread": float(np.ptp(sums) / sums[0]),
            "full_work_band_power_relative_spread": float(np.ptp(full_powers) / full_powers[0]),
            "phase_logparameter_jacobian_singular_values": singular_values.tolist(),
            "amplitude_zero_vs_held_max_change_over_peak": amp_shape_difference,
            "shaft_rate_50_vs_200_max_change_over_peak": shaft_difference,
        },
        "accepted_fit_records": fit_records,
    }
    (OUT / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
