"""Noise model v4 flight fit (``experiments.noise_model.flight_v4``).

The v4 forward model must reduce to the nv3r4 ``flight_v3`` expected
periodogram when the v4 extras are off (pedestal 0, no motor family, b1 = 0,
Delta G = 0) — the lines rotor by rotor, and the ONE floor as nv3r4's floor
driven by the rotor-mean carrier — and its hand-made gradients (the unit-atom
tangent of the speed
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


def _pool(
    n_rotors: int, n_mics: int = 2, n_frames: int = 6, seed: int = 0, mean_rotor: bool = False
) -> MD.SupportBatch:
    """A moving-carrier pool; ``mean_rotor`` replaces the rotors by ONE rotor on
    their mean carrier."""
    n = N_FFT + (n_frames - 1) * HOP
    t = np.linspace(0.0, 1.0, n)
    rps = np.stack([60.0 + 7.0 * r + 25.0 * t + 2.0 * np.sin(9.0 * t + r) for r in range(n_rotors)])
    if mean_rotor:
        rps = rps.mean(axis=0, keepdims=True)
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
    a_sigma: float = 0.0,
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
        # b0 = 2.5 and b1 in the fit's band-edge coordinates (b >= 0 on the band)
        **dict(zip(("u_blo", "u_bhi"), map(np.asarray, V4.floor_coords(2.5, b1)))),
        z=rng.normal(0.0, 0.7, NC),
        log_s=np.asarray(log_s),
        log_sigma_nu=np.asarray(math.log(3.0)),
        a_sigma=np.asarray(a_sigma),
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


def _v3(
    model: FV.FlightV4Model, batch: MD.SupportBatch, sigma_nu: float | None = None
) -> torch.Tensor:
    """The nv3r4 ``flight_v3`` expected periodogram at the model's values."""
    v = {k: t.detach() for k, t in model.values().items()}
    r, m = batch.n_rotors, batch.n_mics
    par = SP.V2Params(
        sigma_nu=v["sigma_nu"] if sigma_nu is None else _t(sigma_nu),
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


def test_one_floor_is_nv3r4_floor_on_the_rotor_mean_carrier():
    """Four rotors on different moving carriers, lines silenced, b1 = 0, no
    static share: the rig's one floor is nv3r4's floor of one rotor riding the
    rotor-mean carrier ``mean_r f_r(t)`` (not a sum of rotor floors, not the
    mean of ``s_r^b``)."""
    batch = _pool(4)
    model = _model(batch, floor_mean_db=-20.0)
    ref = _model(_pool(4, mean_rotor=True), floor_mean_db=-20.0)
    with torch.no_grad():
        model.params["p_a"].fill_(-300.0)
        ref.params["p_a"].fill_(-300.0)
        ref.params["z"].copy_(model.params["z"])
        v4 = model.forward(batch)
    v3 = _v3(ref, _pool(4, mean_rotor=True))
    rel = (v4 - v3).abs() / v3
    assert float(rel.max()) < 1e-6


def test_shaft_law_is_per_rotor_and_frame():
    """``sigma_nu = sigma_ref sbar_rn^a_sigma``: with one rotor sounding, each
    frame's comb is nv3r4's flat-shaft comb at THAT rotor's frame speed (sbar
    the window-power-weighted geometric mean over f_ref), not the rig's mean
    speed or the clip's."""
    batch = _pool(2, n_frames=4)
    a_sigma = 2.5
    assert batch.rate_work is not None and isinstance(batch.grid, SP.FlightGrid)
    w2 = batch.grid.window_work**2
    log_s = torch.log(batch.rate_work / SP.AMP_RPS_REF)
    sbar = torch.exp((log_s * w2).sum(-1) / w2.sum())  # (R, N)
    for r in range(2):
        model = _model(batch, floor_mean_db=-300.0, a_sigma=a_sigma)
        with torch.no_grad():
            model.params["p_a"][1 - r].fill_(-300.0)
            v4 = model.forward(batch)
        for n in range(int(batch.power.shape[1])):
            sub = MD.batch_slice(batch, np.array([n]))
            v3 = _v3(model, sub, sigma_nu=3.0 * float(sbar[r, n]) ** a_sigma)
            live = v3 > 1e-9
            rel = ((v4[:, n : n + 1] - v3).abs() / v3)[live]
            assert float(rel.max()) < 1e-6


@pytest.mark.parametrize(
    "name", ["alpha_a", "alpha_m", "u_blo", "u_bhi", "log_sigma_nu", "a_sigma", "log_s"]
)
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
        a_sigma=1.5,
    )
    for p in model.params.values():
        p.grad = None
    total = float(model.objective(batch, grad=True))
    p = model.params[name]
    assert p.grad is not None
    g = float(p.grad)
    h = 1e-3
    with torch.no_grad():
        x0 = float(p)
        p.fill_(x0 + h)
        up = float(model.objective(batch, grad=False))
        p.fill_(x0 - h)
        dn = float(model.objective(batch, grad=False))
        p.fill_(x0)
    fd = (up - dn) / (2.0 * h)
    # truncation O(h^2) plus the roundoff of a ~5e6-nat sum over cells, / h
    assert abs(g - fd) <= 1e-4 * max(1.0, abs(fd)) + 1e-13 * abs(total) / h
