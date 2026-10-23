"""The similarity battery's two load-bearing properties, on planted audio:
identical inputs read zero everywhere, and a level change confined to the
rotor lines moves the LINE column but not the FLOOR column."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.noise_model import similarity as S

SR = S.SR
T = 2 * SR
RPS = (77.0, 83.0)


def _render(line_gain: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    t = np.arange(T) / SR
    x = 0.01 * rng.standard_normal((2, T))
    for f in RPS:
        for k in range(1, 40):
            phase = rng.uniform(0.0, 2.0 * np.pi, (2, 1))
            x = x + line_gain * 0.004 / np.sqrt(k) * np.cos(2.0 * np.pi * k * f * t + phase)
    return x, np.stack([np.full(T, f) for f in RPS])


def test_identical_inputs_read_zero():
    x, lab = _render(1.0)
    for name, res in S.battery(x, x.copy(), lab).items():
        vals = np.asarray(list(res["summary"].values()), dtype=float)
        fin = vals[np.isfinite(vals)]
        assert fin.size, name
        np.testing.assert_allclose(fin, 0.0, atol=1e-9, err_msg=name)


def test_line_only_offset_moves_line_column_not_floor():
    real, lab = _render(1.0)
    arm, _ = _render(2.0)  # same noise draw and phases, lines +6 dB
    lsd = S.line_floor_lsd(real, arm, lab)["summary"]
    assert lsd["line_signed_db"] > 3.0
    assert lsd["line_abs_db"] == pytest.approx(lsd["line_signed_db"], abs=0.2)
    assert abs(lsd["floor_abs_db"]) < 0.5
    order = S.order_level_error(real, arm, lab)["summary"]
    assert order["peak_abs_db_k1-2"] == pytest.approx(20.0 * np.log10(2.0), abs=0.5)
