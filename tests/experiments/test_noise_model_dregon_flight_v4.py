"""The DREGON flight stage (``experiments.noise_model.dregon_flight_v4``).

Its torch forward model must BE the renderer's: at random floor, static-share,
wind and per-rotor line-gain values, the gained frozen per-rotor bench lines
plus the stage's window transform of the floor terms equal
``v4.expected_periodogram`` of the exported flight payload (gains written into
the profiles) on every bin, with the gains free or held. CPU-small (n_fft 256,
three frames).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.noise_model import dregon_flight_v4 as DF

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "results/noise_v4/bench/restarts/dregon__kappa0__r5__s2.json"
N_FFT, SR = 256, 16000


@pytest.mark.skipif(not BENCH.exists(), reason="needs the tracked DREGON v4 bench payload")
def test_stage_expectation_is_the_renderers() -> None:
    bench = json.loads(BENCH.read_text())
    rps = np.array([[52.0, 71.0, 89.0], [55.0, 70.0, 86.0], [50.0, 74.0, 90.0], [57.0, 68.0, 84.0]])
    n = rps.shape[1]
    f = np.fft.rfftfreq(N_FFT, d=1.0 / SR)
    pool = DF.FlightPool(
        names=("synthetic",),
        power=np.random.default_rng(0).standard_exponential((8, n, f.size)) * 1e-6,
        rps=rps,
        weights=np.ones(n),
        frame_window=np.zeros(n, dtype=np.int64),
        frame_start=np.arange(n) * N_FFT,
        freqs_hz=f,
        sr=SR,
        n_fft=N_FFT,
        hop=N_FFT,
        stride=1,
        n_frames_read=n,
        exposure_scale=1.0,
    )
    lines = DF.frozen_lines(bench, rps, n_fft=N_FFT, sr=SR)  # asserts sum_r L_r = L
    assert lines.shape == (4, 8, n, f.size)
    floor = bench["params"]["floor"]
    # a floor 20 dB over the bench's, so the floor terms carry much of E
    init = dict(
        floor_mean_db=floor["mean_db"] + 20.0,
        floor_z=floor["shape_z"],
        u_blo=0.5,
        u_bhi=0.5,
        log_static=-6.0,
        wind_db=np.full(8, -30.0),
    )
    common = dict(
        band_hz=tuple(bench["band_hz"]),
        kappa=0.0,
        floor_centre_db=floor["mean_db"],
        wind_centre_db=np.full(8, -30.0),
    )
    free = DF.FlightStageModel(
        bench, pool, lines, init=dict(init, line_gain_db=[4.0, -3.0, 7.0, 1.0]), **common
    )
    held = DF.FlightStageModel(bench, pool, lines, init=init, line_gain=False, **common)
    assert DF.GAIN_KEY in free.params and DF.GAIN_KEY not in held.params
    for model in (free, held):
        for seed in (0, 1, 2):
            out = DF.renderer_check(model, [0, 1, 2], seed=seed)
            assert out["max_rel_err"] < DF.CHECK_RTOL, out
            assert np.all(np.abs(out["at"]["line_gain_db"]) > 0.0)
    # the check restores the gains; the payload carries them in the profiles
    np.testing.assert_array_equal(free.line_gain_db.detach().numpy(), [4.0, -3.0, 7.0, 1.0])
    np.testing.assert_array_equal(held.line_gain_db.detach().numpy(), np.zeros(4))
    fl, wind, gain = free.exported()
    pay = DF.flight_payload(bench, floor=fl, wind_db=wind, line_gain_db=gain)
    for key in ("profile_aero_db", "profile_motor_db"):
        diff = np.asarray(pay["params"][key]) - np.asarray(bench["params"][key])
        np.testing.assert_allclose(diff, np.broadcast_to(np.array(gain)[:, None], diff.shape))
