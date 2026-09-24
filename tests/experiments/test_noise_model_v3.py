"""Noise model v3 (``docs/explainers/noise-model-v3-wander.qmd``): its priors,
its data normalisation, its block wander and a synthetic recovery.

Each test pins a consumer-visible property of the ``flight_v3`` mode against
something derived independently of the implementation: the half-normal
density in closed form, the explicit Gaussian of an OU chain, the v2 forward
model bit for bit, or a planted rig. CPU-small by construction.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pyro
import pyro.distributions as dist
import torch

from data_processing.noise_model.floor import floor_geometry
from experiments.noise_model import fit as FT
from experiments.noise_model import model as MD
from experiments.noise_model import spectrum as SP

SR = 16000
NC = SP.FLOOR_SHAPE_N_CTRL


def _t(v: Any) -> torch.Tensor:
    return torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=torch.float64)


def _params(
    *,
    k_cap: int,
    n_mics: int,
    profile_db: Any,
    shape_z: Any = None,
    floor_mean_db: float = -40.0,
    tilt: float = 0.0,
) -> SP.V2Params:
    prof = np.broadcast_to(np.asarray(profile_db, dtype=np.float64), (1, k_cap)).copy()
    return SP.V2Params(
        sigma_nu=_t(0.3),
        lam=_t(0.5),
        gamma_hz=_t(np.full((1, k_cap), 0.02)),
        profile_db=_t(prof),
        floor=SP.FloorParams(
            mean_db=_t(floor_mean_db),
            shape_z=_t(np.zeros(NC) if shape_z is None else shape_z),
            tilt_db_oct=_t(tilt),
            mic_floor_db=_t(np.zeros(n_mics)),
            exp=_t(2.0),
            static_rel=_t(2.5e-3),
        ),
        mic_line_gain_db=_t(np.zeros((n_mics, 1))),
        gain_all_db=_t(np.zeros(n_mics)),
        amp_exp=_t(2.0),
    )


def _flight_batch(
    par: SP.V2Params,
    *,
    k_cap: int,
    n_mics: int,
    n_frames: int = 16,
    seed: int = 3,
    f0: float = 180.0,
    **kw: Any,
) -> MD.SupportBatch:
    """A one-window flight batch whose cells are the model's own ``M`` times
    unit exponentials — the Whittle likelihood's exact generative law."""
    n_fft, hop = 512, 256
    n = n_fft + (n_frames - 1) * hop
    rps = np.full((1, n), f0)
    grid = SP.flight_grid(sr=SR, n_fft=n_fft, hop=hop)
    starts = np.arange(n_frames) * hop
    with torch.no_grad():
        truth = SP.flight_model(
            grid, par, rate_work=SP.flight_rate_work(grid, rps, starts), k_max=k_cap
        ).numpy()
    obs = truth * np.random.default_rng(seed).standard_exponential(truth.shape)
    return MD.flight_batch(
        name="v3",
        members=[("w0", obs[:n_mics], rps, starts)],
        sr=SR,
        n_fft=n_fft,
        hop=hop,
        k_cap=k_cap,
        **kw,
    )


def _measured(batch: MD.SupportBatch, priors: MD.PriorsV3 = MD.PRIORS_V3) -> MD.SupportBatch:
    return dataclasses.replace(
        batch, measured=FT.measure_batch(batch, mode=MD.V3_MODE, priors=priors)
    )


def _trace(batch: MD.SupportBatch, priors: MD.PriorsV3 = MD.PRIORS_V3) -> Any:
    return pyro.poutine.trace(
        lambda: MD.sample_params(batch, mode=MD.V3_MODE, priors=priors)
    ).get_trace()


def _sites(trace: Any) -> set[str]:
    return {
        name
        for name, node in trace.nodes.items()
        if node.get("type") == "sample" and not node.get("is_observed", False)
    }


# ── (a) the dynamics priors ─────────────────────────────────────────────────


def test_line_width_and_shaft_priors_are_light_tailed_in_linear_units():
    """A 13 Hz half width at ``k = 1`` must be ruled out, a bench-law one free.

    ``gamma_rk / (0.01 k) ~ HalfNormal(3)`` charges ``g^2 / (2 s^2)`` with
    ``s = 0.03 k`` Hz: 13 Hz at k = 1 costs ~ 9.4e4 nats, where v2's
    ``LN(log 0.01 k, 1)`` charged ~ 40 and let R4's fit park there. A 0.05 Hz
    line — five bench-law medians — must stay nearly free.
    """
    k_cap = 4
    batch = _measured(
        _flight_batch(_params(k_cap=k_cap, n_mics=2, profile_db=-20.0), k_cap=k_cap, n_mics=2)
    )
    tr = _trace(batch)
    gamma_fn = tr.nodes["gamma_hz"]["fn"].base_dist
    assert isinstance(gamma_fn, dist.HalfNormal)

    def cost(value_hz: float) -> float:
        g = torch.full((1, k_cap), float(value_hz), dtype=torch.float64)
        return -float(gamma_fn.log_prob(g)[0, 0])

    assert cost(13.0) > 1e4
    assert cost(0.05) < 5.0
    # the scale is the bench decoherence law times c_gamma, order by order
    # the scale is the bench decoherence law times c_gamma, order by order
    np.testing.assert_allclose(gamma_fn.scale.numpy()[0], 3.0 * 0.01 * np.arange(1, k_cap + 1))
    sigma_fn = tr.nodes["sigma_nu"]["fn"]
    assert isinstance(sigma_fn, dist.HalfNormal)
    assert float(sigma_fn.scale) == 0.6
    # closed-form half-normal: log(sqrt(2/pi)/s) - x^2 / (2 s^2)
    x = torch.tensor(1.7, dtype=torch.float64)
    want = math.log(math.sqrt(2.0 / math.pi) / 0.6) - 1.7**2 / (2.0 * 0.6**2)
    assert abs(float(sigma_fn.log_prob(x)) - want) < 1e-12


# ── (b) the profile prior ───────────────────────────────────────────────────


def test_every_order_gets_the_same_profile_prior_no_visibility_switch():
    """Visible or not, a line's prior is ``N(pooled level at k f_r, 10 dB)``.

    Half of the planted comb is silent (-300 dB): v2 would park those orders
    at ``N(floor - 15, 8)``. v3 centres them on the pooled level at their own
    frequency, like every other order, with the same 10 dB sd.
    """
    k_cap = 6
    prof = np.array([-15.0, -300.0, -18.0, -300.0, -22.0, -300.0])
    # lines 400 Hz = 13 bins apart, so no order's window reaches a neighbour
    par = _params(k_cap=k_cap, n_mics=2, profile_db=prof)
    batch = _measured(_flight_batch(par, k_cap=k_cap, n_mics=2, f0=400.0))
    meas = batch.measured
    assert meas is not None
    silent = prof[None, :] < -100.0
    # the silent orders really are invisible: v2's switch would have fired
    assert np.all(meas.line_snr_db[silent] < MD.PRIORS.line_visible_snr_db)

    base = _trace(batch).nodes["profile_db"]["fn"].base_dist
    assert isinstance(base, dist.Normal)
    np.testing.assert_array_equal(base.loc.numpy(), np.asarray(meas.profile_db))
    np.testing.assert_array_equal(base.scale.numpy(), np.full((1, k_cap), 10.0))
    loc = np.asarray(meas.profile_db)
    # v2 would have parked the silent orders at N(mu - 15, 8); v3 does not
    park = meas.floor_mean_db + MD.PRIORS.profile_below_offset_db
    assert np.all(np.abs(loc[silent] - park) > 1.0)
    # the silent orders all measure the same thing: the floor under them
    assert float(np.ptp(loc[silent])) < 3.0
    # and a visible one's centre is an UPPER bound on its line: line + floor
    visible = ~silent
    assert np.all(loc[visible] >= prof[None, :][visible] - 1.5)


# ── (c) the floor ───────────────────────────────────────────────────────────


def test_floor_is_the_spline_alone_scaled_by_the_measured_sigma_b():
    """No mean or tilt site; control values ``mu + sigma_B (L z)``.

    A floor planted with a known shape must yield a measured ``sigma_B`` equal
    to the RMS of its control values about the measured level, and the
    forward model's floor must be exactly ``mu + sigma_B L z`` on the spline.
    """
    k_cap = 3
    z_true = np.random.default_rng(5).standard_normal(NC) * 1.5
    par = _params(k_cap=k_cap, n_mics=2, profile_db=-60.0, shape_z=z_true, floor_mean_db=-35.0)
    batch = _measured(_flight_batch(par, k_cap=k_cap, n_mics=2, n_frames=48, seed=9))
    meas = batch.measured
    assert meas is not None and meas.floor_shape_sd_db is not None

    grid = batch.grid
    assert isinstance(grid, SP.FlightGrid)
    c_true = SP.FLOOR_SHAPE_STD_DB * (grid.floor.shape_chol.numpy() @ z_true)
    basis = floor_geometry(grid.freqs_hz[grid.band], grid.floor.ctrl_hz)[0]
    c_rel = c_true - np.median(basis @ c_true)
    want = float(np.sqrt(np.mean(c_rel**2)))
    assert abs(meas.floor_shape_sd_db - want) < 1.0, (meas.floor_shape_sd_db, want)

    tr = _trace(batch)
    sites = _sites(tr)
    assert "floor_shape_z" in sites
    assert sites.isdisjoint({"floor_mean_db", "floor_tilt_db_oct"})

    z = torch.as_tensor(np.random.default_rng(1).standard_normal(NC))
    values = {name: tr.nodes[name]["value"] for name in sites}
    values["floor_shape_z"] = z
    p = MD.sample_params_from_values(batch, mode=MD.V3_MODE, priors=MD.PRIORS_V3, values=values)
    assert float(p.floor.mean_db) == meas.floor_mean_db
    assert float(p.floor.shape_sd_db) == meas.floor_shape_sd_db
    assert float(p.floor.tilt_db_oct) == 0.0
    want_ctrl = meas.floor_shape_sd_db * (grid.floor.shape_chol @ z)
    got_ctrl = grid.floor.shape_db(p.floor.shape_z, p.floor.shape_sd_db)
    assert torch.equal(got_ctrl, want_ctrl)
    psd = grid.floor.psd(p.floor)
    db = meas.floor_mean_db + grid.floor.shape_psd @ want_ctrl
    want_psd = grid.floor.rate_factor * 10.0 ** (db / 10.0)
    assert float(((psd - want_psd).abs() / want_psd).max()) < 1e-12
