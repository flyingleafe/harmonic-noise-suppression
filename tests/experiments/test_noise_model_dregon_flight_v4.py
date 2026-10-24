"""The DREGON flight stage (``experiments.noise_model.dregon_flight_v4``).

Its torch forward model must BE the renderer's: at random floor, static-share,
wind, per-rotor line-gain and per-mic line-gain values, the gained frozen
per-rotor bench lines plus the stage's window transform of the floor terms
equal ``v4.expected_periodogram`` of the exported flight payload (rotor gains
written into the profiles, mic gains into the pair EQ) on every bin, with the
gains free or held; and with the line coherence free (``coherence="shaft"``:
``eps`` only, the bench phase pedestal held; ``"full"``: pedestal and ``eps``),
the torch line renderer at a random ``eps`` (and pedestal) too. A FLIGHT source
(the stage's or the wind stage's payload) starts at its own values, where the
stage's ``E`` is the source's expected periodogram (its ``sc`` wind shape
included), and exports back to itself; ``coherence="lines"`` (free low-order
profiles and pedestal) is the renderer's at random values and leaves the orders
above ``k_free`` as the source's. CPU-small (n_fft 256, three frames).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data_processing.noise_model import v4 as V4
from experiments.noise_model import dregon_flight_v4 as DF

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "results/noise_v4/bench/restarts/dregon__kappa0__r5__s2.json"
#: the static-wind stage payload and the C-wind (sc) payload built on it
STAGE = ROOT / "results/noise_v4/flight/dregon_room2__flight_v4__dregon__kappa0__r5__s2.json"
SC = ROOT / (
    "results/noise_v4/flight/wind/dregon_room2__flight_v4__dregon__kappa0__r5__s2__scwind.json"
)
N_FFT, SR = 256, 16000
RPS = np.array([[52.0, 71.0, 89.0], [55.0, 70.0, 86.0], [50.0, 74.0, 90.0], [57.0, 68.0, 84.0]])
LINE_GAIN = [4.0, -3.0, 7.0, 1.0]
MIC_GAIN = [2.0, -1.0, 5.0, -4.0, 0.5, -3.5, 6.0, -5.0]  # sums to 0


def _pool(rps: np.ndarray) -> DF.FlightPool:
    n = rps.shape[1]
    f = np.fft.rfftfreq(N_FFT, d=1.0 / SR)
    return DF.FlightPool(
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


def _init(bench: dict) -> dict:
    floor = bench["params"]["floor"]
    # a floor 20 dB over the bench's, so the floor terms carry much of E
    return dict(
        floor_mean_db=floor["mean_db"] + 20.0,
        floor_z=floor["shape_z"],
        u_blo=0.5,
        u_bhi=0.5,
        log_static=-6.0,
        wind_db=np.full(8, -30.0),
    )


def _common(bench: dict) -> dict:
    return dict(
        band_hz=tuple(bench["band_hz"]),
        kappa=0.0,
        floor_centre_db=bench["params"]["floor"]["mean_db"],
        wind_centre_db=np.full(8, -30.0),
    )


@pytest.mark.skipif(not BENCH.exists(), reason="needs the tracked DREGON v4 bench payload")
def test_stage_expectation_is_the_renderers() -> None:
    bench = json.loads(BENCH.read_text())
    rps, n = RPS, RPS.shape[1]
    pool = _pool(rps)
    lines = DF.frozen_lines(bench, rps, n_fft=N_FFT, sr=SR)  # asserts sum_r L_r = L
    assert lines.shape == (4, 8, n, N_FFT // 2 + 1)
    init, common = _init(bench), _common(bench)
    free = DF.FlightStageModel(
        bench, pool, lines, init=dict(init, line_gain_db=LINE_GAIN, mic_gain_db=MIC_GAIN), **common
    )
    held = DF.FlightStageModel(
        bench, pool, lines, init=init, line_gain=False, mic_gain=False, **common
    )
    assert DF.GAIN_KEY in free.params and DF.GAIN_KEY not in held.params
    assert DF.MIC_GAIN_KEY in free.params and DF.MIC_GAIN_KEY not in held.params
    for model in (free, held):
        for seed in (0, 1, 2):
            out = DF.renderer_check(model, [0, 1, 2], seed=seed)
            assert out["max_rel_err"] < DF.CHECK_RTOL, out
            assert np.all(np.abs(out["at"]["line_gain_db"]) > 0.0)
            assert np.all(np.abs(out["at"]["mic_gain_db"]) > 0.0)
            assert abs(sum(out["at"]["mic_gain_db"])) < 1e-12
    # the check restores the gains; the payload carries them in the profiles and pair EQ
    np.testing.assert_array_equal(free.line_gain_db.detach().numpy(), LINE_GAIN)
    np.testing.assert_array_equal(held.line_gain_db.detach().numpy(), np.zeros(4))
    np.testing.assert_allclose(free.values()["mic_gain_db"].detach().numpy(), MIC_GAIN)
    np.testing.assert_array_equal(held.values()["mic_gain_db"].detach().numpy(), np.zeros(8))
    fl, wind, gain, mic = free.exported()
    pay = DF.flight_payload(bench, floor=fl, wind_db=wind, line_gain_db=gain, mic_gain_db=mic)
    for key in ("profile_aero_db", "profile_motor_db"):
        diff = np.asarray(pay["params"][key]) - np.asarray(bench["params"][key])
        np.testing.assert_allclose(diff, np.broadcast_to(np.array(gain)[:, None], diff.shape))
    eq_pay = np.asarray(pay["params"]["mic_response"]["pair_eq"]["db"])
    eq_bench = np.asarray(bench["params"]["mic_response"]["pair_eq"]["db"])
    diff = eq_pay - eq_bench
    np.testing.assert_allclose(diff, np.broadcast_to(np.array(mic)[None, :, None], diff.shape))


@pytest.mark.skipif(not BENCH.exists(), reason="needs the tracked DREGON v4 bench payload")
@pytest.mark.parametrize("coherence", ["shaft", "full"])
def test_coherence_lines_are_the_renderers(coherence: str) -> None:
    bench = json.loads(BENCH.read_text())
    model = DF.FlightStageModel(
        bench,
        _pool(RPS),
        None,
        init=dict(_init(bench), line_gain_db=LINE_GAIN, mic_gain_db=MIC_GAIN),
        coherence=coherence,
        **_common(bench),
    )
    free = DF.COHERENCE_FREE[coherence]
    assert [k for k in DF.COHERENCE_KEYS if k in model.params] == list(free)
    start = {k: model.params[k].detach().clone() for k in free}
    for seed in (0, 1):
        out = DF.renderer_check(model, [0, 1, 2], seed=seed)
        assert out["max_rel_err"] < DF.CHECK_RTOL, out
        assert np.all(np.abs(out["at"]["mic_gain_db"]) > 0.0)
        # the check ran at a shaft scale other than the bench's
        assert out["at"]["coherence"]["shaft"]["eps"] != bench["params"]["shaft"]["eps"]
        # "shaft" leaves the bench phase pedestal as it is
        assert ("pedestal" in out["at"]["coherence"]) == (coherence == "full")
    # ... and restored the start
    for k, t in start.items():
        np.testing.assert_array_equal(model.params[k].detach().numpy(), t.numpy())


def _at_source(source: dict, **kw) -> DF.FlightStageModel:
    init = DF.source_init(source)
    lines = None if kw.get("coherence") else DF.frozen_lines(source, RPS, n_fft=N_FFT, sr=SR)
    return DF.FlightStageModel(
        source,
        _pool(RPS),
        lines,
        init=init,
        band_hz=tuple(source["band_hz"]),
        kappa=0.0,
        floor_centre_db=init["floor_mean_db"],
        wind_centre_db=init["wind_db"],
        **kw,
    )


def _source_periodogram(source: dict) -> np.ndarray:
    return np.stack(
        [V4.expected_periodogram(source, RPS[:, i], n_fft=N_FFT, sr=SR) for i in range(3)], axis=1
    )


@pytest.mark.skipif(not SC.exists(), reason="needs the tracked DREGON C-wind payload")
def test_an_sc_wind_source_is_the_stage_at_its_own_values() -> None:
    sc = json.loads(SC.read_text())
    model = _at_source(sc)
    assert model.wind_kind == V4.WIND_SC
    mine = model.expected(slice(None), all_bins=True).detach().numpy()
    assert np.max(np.abs(mine / _source_periodogram(sc) - 1.0)) < DF.CHECK_RTOL
    # the export keeps the gust law and its shape; only the levels are the stage's
    fl, wind, gain, mic = model.exported()
    wind = (np.asarray(wind) - 3.0).tolist()
    pay = DF.flight_payload(sc, floor=fl, wind_db=wind, line_gain_db=gain, mic_gain_db=mic)
    src, out = sc["params"]["wind"], pay["params"]["wind"]
    assert out["level_db"] == wind
    assert {k: v for k, v in out.items() if k != "level_db"} == {
        k: v for k, v in src.items() if k != "level_db"
    }
    # and at random values (levels on the sc shape) the stage is the renderer's
    assert DF.renderer_check(model, [0, 1, 2], seed=3)["max_rel_err"] < DF.CHECK_RTOL


@pytest.mark.skipif(not STAGE.exists(), reason="needs the tracked DREGON stage payload")
def test_a_static_wind_source_exports_back_to_itself() -> None:
    stage = json.loads(STAGE.read_text())
    model = _at_source(stage)
    assert model.wind_kind == V4.WIND_STATIC
    fl, wind, gain, mic = model.exported()
    pay = DF.flight_payload(stage, floor=fl, wind_db=wind, line_gain_db=gain, mic_gain_db=mic)
    src, out = stage["params"], pay["params"]
    assert out["wind"] == src["wind"]
    for key in ("profile_aero_db", "profile_motor_db", "pedestal", "shaft", "mic_response"):
        assert out[key] == src[key], key
    for key in ("mean_db", "shape_z", "ctrl_db", "b0", "b1", "static_rel"):
        np.testing.assert_allclose(out["floor"][key], src["floor"][key], rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(not SC.exists(), reason="needs the tracked DREGON C-wind payload")
def test_the_line_refit_is_the_renderers_and_holds_the_high_orders() -> None:
    sc = json.loads(SC.read_text())
    k = 12
    model = _at_source(sc, coherence="lines", line_gain=False, k_free=k)
    assert tuple(model.params[DF.PROFILE_KEY].shape) == (4, k)
    assert DF.GAIN_KEY not in model.params
    for seed in (0, 1):
        out = DF.renderer_check(model, [0, 1, 2], seed=seed)
        assert out["max_rel_err"] < DF.CHECK_RTOL, out
        coh, src = out["at"]["coherence"], sc["params"]
        prof = np.asarray(coh["profile_aero_db"])
        # the free entries moved (prior draws), the orders above k_free did not
        assert np.all(prof[:, :k] != np.asarray(src["profile_aero_db"])[:, :k])
        np.testing.assert_array_equal(prof[:, k:], np.asarray(src["profile_aero_db"])[:, k:])
        for key in ("sigma2_phase", "gamma_phase_hz"):
            np.testing.assert_array_equal(coh["pedestal"][key][k:], src["pedestal"][key][k:])
