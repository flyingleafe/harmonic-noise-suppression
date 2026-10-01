"""Static-rig campaign: the combined comb read and the cross-fitted
point-source test (``docs/experiments/static-rig-profiles.md``)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import static_rig_report as R

from experiments.static_rig import spectra as S


def _spectrum(lines: dict[float, float], floor: float = 1e-6, df: float = 0.0625) -> S.Spectrum:
    """One channel, flat floor density ``floor``, a line of total power ``p``
    in the single bin nearest each ``f``."""
    f = np.arange(1, int(2000 / df)) * df
    psd = np.full((1, f.size), floor)
    for fc, p in lines.items():
        psd[0, int(np.argmin(np.abs(f - fc)))] += p / df
    return S.Spectrum(f=f, psd=psd, floor=psd.copy(), df=df, fs=4000)


def test_comb_band_powers_sums_every_rotor_in_band_and_stops_before_overlap() -> None:
    speeds, p_line = (70.0, 70.8, 71.5, 72.0), 1e-3
    lines = {k * s: p_line for k in range(1, 30) for s in speeds}
    out = S.comb_band_powers(_spectrum(lines), 69.8, 72.2, S.Params())
    w = S.Params().line_bins * 0.0625
    # last order whose band + flanks stay clear of the next order's band
    k_max = max(k for k in range(1, 40) if 69.8 - k * 2.4 >= 2 * (w + 10.0))
    assert out["orders"] == list(range(1, k_max + 1))
    np.testing.assert_allclose(out["power_db"][:, 0], 10 * math.log10(4 * p_line), atol=0.05)


def test_comb_band_powers_floor_comes_from_flanks_not_band() -> None:
    # the band is packed with lines; a running-median floor would sit on them
    out = S.comb_band_powers(_spectrum({100.0: 1e-3}, floor=1e-6), 99.0, 101.0, S.Params())
    band_bins = 2 * 1.0 / 0.0625 + 2 * S.Params().line_bins + 1
    assert out["snr_db"][0, 0] == pytest.approx(
        10 * math.log10((1e-3 + band_bins * 1e-6 * 0.0625) / (band_bins * 1e-6 * 0.0625)), abs=0.1
    )


_ROTORS = np.array([[0.2, 0.2, 0.0], [0.2, -0.2, 0.0], [-0.2, -0.2, 0.0], [-0.2, 0.2, 0.0]])
_MICS = np.array([[0.5, 0.0, -0.1], [0.0, 0.6, -0.2], [-0.4, -0.1, -0.1], [0.1, -0.5, 0.3]])


def _obs(even_rotors: tuple[int, int], odd_rotors: tuple[int, int]) -> dict:
    """Two tracks whose even orders radiate from ``even_rotors`` and odd orders
    from ``odd_rotors`` (pure 1/r, no mic gains)."""
    D = R._dist({"mic_pos": _MICS, "rotor_pos": _ROTORS})
    rotors = []
    for t in range(2):
        orders = list(range(1, 13))
        power = [
            (-10.0 * k - 20 * np.log10(D[(even_rotors if k % 2 == 0 else odd_rotors)[t]])).tolist()
            for k in orders
        ]
        rotors.append(
            {
                "track": [80.0 + 5 * t] * 20,
                "mean": 80.0 + 5 * t,
                "std": 0.1,
                "distinct": True,
                "orders": orders,
                "power_db": power,
                "snr_db": [[30.0] * 4 for _ in orders],
                "phase_rel": [
                    {"ref": 0, "phase": [0.0] * 4, "coherence": [0.0] * 4} for _ in orders
                ],
            }
        )
    return {
        "dataset": "AVQ",
        "key": "synthetic",
        "rig": "avq",
        "n_rotors": 4,
        "rate_range": [60.0, 120.0],
        "residual_ratios": [0.2],
        "rotors": rotors,
    }


@pytest.fixture
def _geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R, "geometry", lambda o: {"mic_pos": _MICS, "rotor_pos": _ROTORS})


@pytest.mark.usefixtures("_geometry")
def test_point_source_recovers_pure_spreading_out_of_sample() -> None:
    res = R.point_source([_obs((0, 2), (0, 2))])["avq"]
    assert res["rms_dist"] == pytest.approx(0.0, abs=1e-6)
    assert res["alpha_dist"] == pytest.approx([1.0, 1.0], abs=1e-3)


@pytest.mark.usefixtures("_geometry")
def test_point_source_permutation_is_chosen_on_the_training_parity_only() -> None:
    # each parity alone is exactly 1/r, but under different track -> rotor
    # assignments; choosing the permutation on the scored parity would hide it
    res = R.point_source([_obs((0, 2), (1, 3))])["avq"]
    assert res["rms_dist"] > 1.0
