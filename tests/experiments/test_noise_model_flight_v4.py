"""Noise model v4 flight fit (``experiments.noise_model.flight_v4``).

The v4 forward model must reduce to the nv3r4 ``flight_v3`` expected
periodogram when the v4 extras are off (pedestal 0, no motor family, b1 = 0,
Delta G = 0), and its hand-made gradients (the unit-atom tangent of the speed
exponents) must be the objective's derivative. CPU-small by construction.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch

from experiments.noise_model import flight_v4 as FV
from experiments.noise_model import model as MD
from experiments.noise_model import spectrum as SP
from experiments.noise_model import v4_priors as V4

SR = 16000
N_FFT, HOP, SR_WORK = 512, 256, 32000
K = 20
NC = SP.FLOOR_SHAPE_N_CTRL


def _t(v: Any) -> torch.Tensor:
    return torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=torch.float64)


def _pool(n_rotors: int, n_mics: int = 2, n_frames: int = 6, seed: int = 0) -> MD.SupportBatch:
    n = N_FFT + (n_frames - 1) * HOP
    t = np.linspace(0.0, 1.0, n)
    rps = np.stack([60.0 + 7.0 * r + 25.0 * t + 2.0 * np.sin(9.0 * t + r) for r in range(n_rotors)])
    starts = np.arange(n_frames) * HOP
    obs = np.random.default_rng(seed).standard_exponential((n_mics, n_frames, N_FFT // 2 + 1))
    return MD.flight_batch(
        name="v4",
        members=[("w0", obs * 1e-6, rps, starts)],
        sr=SR,
        n_fft=N_FFT,
        hop=HOP,
        k_cap=K,
        sr_work=SR_WORK,
        chunk_frames=4,
    )


def _model(
    batch: MD.SupportBatch,
    *,
    pedestal: float = 0.0,
    motor: bool = False,
    geometry: bool = False,
    b1: float = 0.0,
    log_s: float = -200.0,
    floor_mean_db: float = -60.0,
    seed: int = 1,
) -> FV.FlightV4Model:
    rng = np.random.default_rng(seed)
    r, m = batch.n_rotors, batch.n_mics
    motor_orders = np.array([7, 14], dtype=np.int64) if motor else np.zeros(0, dtype=np.int64)
    ped = {
        key: np.full(K, val)
        for key, val in (
            ("sigma2_amp", pedestal),
            ("gamma_amp_hz", 3.0),
            ("sigma2_phase", pedestal),
            ("gamma_phase_hz", 9.0),
            ("sigma2_amp_uncapped", pedestal),
        )
    }
    data = FV.FlightData(
        batch=batch,
        k_max=K,
        motor_orders=motor_orders,
        geometry_db=rng.normal(0.0, 1.0, (r, m)) if geometry else np.zeros((r, m)),
        pedestal=ped,
        wander_path="synthetic",
        mics=list(range(m)),
    )
    p_a = rng.normal(-30.0, 6.0, (r, K))
    p_m = rng.normal(-25.0, 3.0, (r, motor_orders.size))
    init = dict(
        p_a=p_a,
        pbar_a=p_a.mean(axis=0),
        c_a=np.zeros(r - 1),
        p_m=p_m,
        pbar_m=p_m.mean(axis=0),
        c_m=np.zeros(r - 1),
        alpha_a=np.asarray(5.0),
        alpha_m=np.asarray(-0.5),
        b0=np.asarray(2.5),
        b1=np.asarray(b1),
        z=rng.normal(0.0, 0.7, NC),
        log_s=np.asarray(log_s),
        log_sigma_nu=np.asarray(math.log(3.0)),
    )
    meas = FV.FlightMeasured(
        floor_mean_db=floor_mean_db,
        floor_shape_sd_db=4.0,
        floor_ctrl_db=np.zeros(NC),
        floor_line_db=np.full((r, K), -40.0),
        floor_order_db=np.full(K, -40.0),
        init=init,
        record={},
    )
    return FV.FlightV4Model(data, meas, kappa=0.3)


def _v3(model: FV.FlightV4Model, batch: MD.SupportBatch) -> torch.Tensor:
    """The nv3r4 ``flight_v3`` expected periodogram at the model's values."""
    v = {k: t.detach() for k, t in model.values().items()}
    r, m = batch.n_rotors, batch.n_mics
    par = SP.V2Params(
        sigma_nu=v["sigma_nu"],
        lam=_t(V4.FLIGHT_LAM),
        gamma_hz=_t(np.zeros((r, K))),
        profile_db=v["p_a"],
        floor=SP.FloorParams(
            mean_db=model.mu,
            shape_z=v["z"],
            tilt_db_oct=_t(0.0),
            mic_floor_db=_t(np.zeros(m)),
            exp=v["b0"],
            static_rel=_t(0.0),
            shape_sd_db=model.sigma_b,
        ),
        mic_line_gain_db=_t(np.zeros((m, r))),
        gain_all_db=_t(np.zeros(m)),
        amp_exp=v["alpha_a"],
    )
    assert batch.rate_work is not None and isinstance(batch.grid, SP.FlightGrid)
    with torch.no_grad():
        return SP.flight_model(
            batch.grid, par, rate_work=batch.rate_work, k_max=K, unit_autocorr=True
        )


def test_lines_reduce_to_nv3r4_with_the_v4_extras_off():
    """Four rotors, moving carriers: pedestal 0, no motor family, Delta G 0 —
    the comb must be nv3r4's to 1e-6 (the floor is pushed out of the way)."""
    batch = _pool(4)
    model = _model(batch, floor_mean_db=-300.0)
    with torch.no_grad():
        v4 = model.forward(batch)
    v3 = _v3(model, batch)
    live = v3 > 1e-9
    assert live.float().mean() > 0.5
    rel = ((v4 - v3).abs() / v3)[live]
    assert float(rel.max()) < 1e-6


def test_one_rotor_floor_reduces_to_nv3r4():
    """One rotor, b1 = 0, no static share: the per-rotor floor is nv3r4's
    floor (the sum over rotors and the mean over rotors coincide)."""
    batch = _pool(1)
    model = _model(batch, floor_mean_db=-20.0)
    with torch.no_grad():
        v4 = model.forward(batch)
    v3 = _v3(model, batch)
    rel = (v4 - v3).abs() / v3
    assert float(rel.max()) < 1e-6


@pytest.mark.parametrize("name", ["alpha_a", "alpha_m", "b0", "b1", "log_sigma_nu", "log_s"])
def test_objective_gradient_matches_finite_differences(name: str):
    """Every extra on: the accumulated gradient of a scalar site equals the
    central difference of the objective."""
    batch = _pool(3)
    batch.power.mul_(1e6)
    model = _model(
        batch,
        pedestal=0.2,
        motor=True,
        geometry=True,
        b1=0.4,
        log_s=math.log(0.02),
        floor_mean_db=-35.0,
    )
    for p in model.params.values():
        p.grad = None
    model.objective(batch, grad=True)
    p = model.params[name]
    assert p.grad is not None
    g = float(p.grad)
    h = 1e-4
    with torch.no_grad():
        x0 = float(p)
        p.fill_(x0 + h)
        up = float(model.objective(batch, grad=False))
        p.fill_(x0 - h)
        dn = float(model.objective(batch, grad=False))
        p.fill_(x0)
    fd = (up - dn) / (2.0 * h)
    assert abs(g - fd) <= 1e-4 * max(1.0, abs(fd))
